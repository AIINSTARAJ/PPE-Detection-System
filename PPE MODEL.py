
"""
PPE Detection System using YOLOv5

A computer vision system for detecting Personal Protective Equipment in industrial settings.
Supports training, evaluation, and real-time inference via Flask API.

Classes detected: hardhat, goggles, vest, gloves, boots, mask, person
"""

import os
import sys
import yaml
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import base64
import io
import time
from datetime import datetime

# Basic configuration
class Config:
    """Simple configuration for the PPE detection system"""
    
    # Model settings - using YOLOv5 nano for speed
    MODEL_SIZE = 'n'
    INPUT_SIZE = 320  # Image input size (320x320 pixels)
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    EPOCHS = 100
    
    # Detection thresholds
    CONF_THRESH = 0.5  # Minimum confidence to consider a detection
    IOU_THRESH = 0.4   # Overlap threshold for removing duplicate detections
    
    # PPE classes we want to detect
    CLASSES = ["hardhat", "goggles", "vest", "gloves", "boots", "mask", "person"]
    
    # Which PPE items are mandatory for compliance
    REQUIRED_PPE = ["hardhat", "vest", "goggles"]
    
    # Device selection (GPU if available, otherwise CPU)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPETrainer:
    """
    Handles training of the PPE detection model
    
    Takes a dataset folder and trains a YOLOv5 model to detect PPE items.
    Includes data augmentation and saves the best performing model.
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.config = Config()
        
        # Create folder for saving trained models
        self.models_dir = Path("trained_models")
        self.models_dir.mkdir(exist_ok=True)
        
        print(f"Using device: {self.config.DEVICE}")
        print(f"Dataset path: {self.dataset_path}")
    
    def setup_dataset_config(self):
        """
        Creates the YAML configuration file that YOLOv5 needs for training
        
        YOLOv5 expects a specific folder structure:
        - train/images and train/labels
        - valid/images and valid/labels
        - dataset.yaml file describing the setup
        """
        
        # Check if the required folders exist
        required_folders = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
        for folder in required_folders:
            folder_path = self.dataset_path / folder
            if not folder_path.exists():
                raise FileNotFoundError(f"Missing required folder: {folder_path}")
        
        # Create the YAML config that tells YOLOv5 about our dataset
        dataset_config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images' if (self.dataset_path / 'test/images').exists() else 'valid/images',
            'nc': len(self.config.CLASSES),  # number of classes
            'names': self.config.CLASSES     # class names
        }
        
        # Save the config file
        yaml_path = self.dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f)
        
        print(f"Dataset config saved: {yaml_path}")
        return yaml_path
    
    def train(self):
        """
        Main training function
        
        Loads a pre-trained YOLOv5 model and fine-tunes it on our PPE dataset.
        Uses data augmentation to improve model generalization.
        """
        
        # Install YOLOv5 if not available
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Installing YOLOv5...")
            os.system("pip install ultralytics")
            from ultralytics import YOLO
        
        # Setup dataset configuration
        yaml_path = self.setup_dataset_config()
        
        # Load pre-trained model (trained on COCO dataset)
        # This gives us a head start instead of training from scratch
        model_name = f'yolov5{self.config.MODEL_SIZE}u.pt'
        print(f"Loading pre-trained model: {model_name}")
        model = YOLO(model_name)
        
        # Training configuration with data augmentation
        # Data augmentation helps the model generalize better by showing it
        # variations of the training images (rotated, scaled, color-shifted, etc.)
        train_config = {
            'data': str(yaml_path),
            'epochs': self.config.EPOCHS,
            'imgsz': self.config.INPUT_SIZE,
            'batch': self.config.BATCH_SIZE,
            'device': self.config.DEVICE,
            'workers': 4,
            'project': 'training_runs',
            'name': 'ppe_model',
            'save_period': 20,  # Save checkpoint every 20 epochs
            'cache': True,      # Cache images in memory for faster training
            
            # Optimizer settings
            'optimizer': 'AdamW',
            'lr0': self.config.LEARNING_RATE,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            
            # Data augmentation parameters
            # These create variations of training images to improve generalization
            'hsv_h': 0.015,    # Slight hue changes
            'hsv_s': 0.7,      # Saturation changes
            'hsv_v': 0.4,      # Brightness changes
            'degrees': 10,     # Small rotations (±10 degrees)
            'translate': 0.1,  # Small position shifts
            'scale': 0.5,      # Size variations
            'fliplr': 0.5,     # Horizontal flips (50% chance)
            'mosaic': 1.0,     # Mosaic augmentation (combines 4 images)
            'mixup': 0.1,      # MixUp augmentation
        }
        
        print("Starting training...")
        print(f"Training for {self.config.EPOCHS} epochs")
        
        # Start training
        start_time = time.time()
        results = model.train(**train_config)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time/60:.1f} minutes")
        
        # Copy the best model to our models directory for easy access
        best_model_path = Path('training_runs/ppe_model/weights/best.pt')
        if best_model_path.exists():
            import shutil
            final_path = self.models_dir / 'best_ppe_model.pt'
            shutil.copy(best_model_path, final_path)
            print(f"Best model saved to: {final_path}")
        
        return results
    
    def evaluate(self, model_path=None):
        """
        Evaluates the trained model performance
        
        Runs the model on validation data and calculates metrics like:
        - mAP (mean Average Precision): How accurate the detections are
        - Precision: Of all detections, how many were correct
        - Recall: Of all actual objects, how many were detected
        - F1-score: Balance between precision and recall
        """
        
        if model_path is None:
            model_path = self.models_dir / 'best_ppe_model.pt'
        
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            return None
        
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        
        # Run validation
        yaml_path = self.dataset_path / 'dataset.yaml'
        print("Evaluating model...")
        results = model.val(data=str(yaml_path))
        
        # Extract key metrics
        metrics = {
            'mAP_50': float(results.box.map50),      # mAP at 50% overlap threshold
            'mAP_50_95': float(results.box.map),     # mAP averaged over 50-95% thresholds
            'precision': float(results.box.mp),      # Average precision across classes
            'recall': float(results.box.mr),         # Average recall across classes
        }
        
        # Calculate F1-score (harmonic mean of precision and recall)
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        print("\nModel Performance:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric:12s}: {value:.4f}")
        print("-" * 30)
        
        return metrics

class PPEDetector:
    """
    Handles inference (prediction) using a trained PPE detection model
    
    Loads a trained model and can detect PPE items in new images.
    Also calculates compliance based on which PPE items are detected.
    """
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.config = Config()
        
        # Statistics tracking
        self.detection_count = 0
        self.total_inference_time = 0
        
        # Load the trained model
        self.load_model()
    
    def load_model(self):
        """Load the trained YOLOv5 model"""
        
        try:
            from ultralytics import YOLO
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = YOLO(str(self.model_path))
            self.model_loaded = True
            print(f"Model loaded: {self.model_path}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            self.model_loaded = False
    
    def preprocess_image(self, image, target_size=None):
        """
        Prepares an image for model input
        
        Takes an image in various formats and converts it to the format
        expected by the model (RGB, specific size, etc.)
        """
        
        if target_size is None:
            target_size = self.config.INPUT_SIZE
        
        # Convert different input types to PIL Image
        if isinstance(image, np.ndarray):
            # Handle OpenCV images (BGR format)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Ensure RGB format
        image = image.convert('RGB')
        
        # Resize while keeping aspect ratio
        # This prevents distortion that could hurt detection accuracy
        w, h = image.size
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        
        # Use high-quality resampling
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return image
    
    def detect(self, image):
        """
        Main detection function
        
        Takes an image and returns all detected PPE items with their locations,
        confidence scores, and compliance analysis.
        """
        
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        start_time = time.time()
        
        try:
            # Prepare image for model input
            processed_image = self.preprocess_image(image)
            
            # Run detection
            results = self.model(
                processed_image,
                conf=self.config.CONF_THRESH,
                iou=self.config.IOU_THRESH,
                verbose=False
            )
            
            # Process results
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Extract detection information
                    class_id = int(box.cls)
                    class_name = self.config.CLASSES[class_id]
                    confidence = float(box.conf)
                    
                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': {
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                    }
                    detections.append(detection)
            
            # Calculate compliance
            compliance = self.check_compliance(detections)
            
            # Update statistics
            inference_time = time.time() - start_time
            self.detection_count += 1
            self.total_inference_time += inference_time
            
            return {
                'success': True,
                'detections': detections,
                'compliance': compliance,
                'inference_time_ms': inference_time * 1000,
                'total_detections': len(detections)
            }
            
        except Exception as e:
            return {"error": f"Detection failed: {str(e)}", "success": False}
    
    def check_compliance(self, detections):
        """
        Analyzes detected items to determine PPE compliance
        
        Checks if required PPE items are present and calculates a compliance score.
        In industrial settings, this helps identify safety violations.
        """
        
        # Count people in the image
        people_count = sum(1 for d in detections if d['class'] == 'person')
        
        if people_count == 0:
            return {
                'status': 'no_person',
                'score': 0,
                'message': 'No person detected in image'
            }
        
        # Find all detected PPE items (excluding 'person')
        detected_ppe = set()
        for detection in detections:
            if detection['class'] != 'person':
                detected_ppe.add(detection['class'])
        
        # Check which required PPE items are present
        required_set = set(self.config.REQUIRED_PPE)
        present_required = required_set.intersection(detected_ppe)
        missing_required = required_set - detected_ppe
        
        # Calculate compliance score (percentage of required PPE present)
        compliance_score = len(present_required) / len(required_set)
        
        # Determine overall compliance status
        is_compliant = compliance_score >= 0.8  # 80% threshold
        status = 'compliant' if is_compliant else 'non_compliant'
        
        return {
            'status': status,
            'score': compliance_score,
            'people_count': people_count,
            'required_ppe': list(required_set),
            'detected_ppe': list(detected_ppe),
            'missing_ppe': list(missing_required),
            'is_compliant': is_compliant
        }
    
    def get_stats(self):
        """Returns detection statistics"""
        avg_time = self.total_inference_time / self.detection_count if self.detection_count > 0 else 0
        return {
            'total_detections': self.detection_count,
            'average_time_ms': avg_time * 1000,
            'total_time_s': self.total_inference_time
        }

# Flask API for real-time detection
app = Flask(__name__)
detector = None

def init_api(model_path):
    """Initialize the detection API with a trained model"""
    global detector
    detector = PPEDetector(model_path)
    return detector.model_loaded

@app.route('/health', methods=['GET'])
def health():
    """Check if the API is running and model is loaded"""
    return jsonify({
        'status': 'running',
        'model_loaded': detector is not None and detector.model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/detect', methods=['POST'])
def api_detect():
    """
    Main detection endpoint
    
    Expects JSON with base64-encoded image:
    {
        "image": "base64_encoded_image_data",
        "conf_threshold": 0.5  // optional
    }
    
    Returns detection results and compliance analysis.
    """
    
    try:
        if not detector or not detector.model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode the image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({'error': f'Invalid image: {str(e)}'}), 400
        
        # Update confidence threshold if provided
        if 'conf_threshold' in data:
            detector.config.CONF_THRESH = float(data['conf_threshold'])
        
        # Run detection
        results = detector.detect(image)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_simple', methods=['POST'])
def api_detect_simple():
    """
    Simplified endpoint for embedded systems (like ESP32)
    
    Returns minimal response to reduce data transfer:
    {
        "compliant": true/false,
        "score": 0.85,
        "missing": ["hardhat"]
    }
    """
    
    try:
        if not detector or not detector.model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Handle both JSON and raw image data
        if request.content_type == 'application/json':
            data = request.get_json()
            image_data = base64.b64decode(data['image'])
        else:
            image_data = request.data
        
        # Load image
        try:
            image = Image.open(io.BytesIO(image_data))
        except:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Run detection with higher confidence for embedded use
        detector.config.CONF_THRESH = 0.6
        results = detector.detect(image)
        
        if not results['success']:
            return jsonify({'error': results.get('error', 'Detection failed')}), 500
        
        # Return simplified response
        compliance = results['compliance']
        return jsonify({
            'compliant': compliance['is_compliant'],
            'score': round(compliance['score'], 2),
            'missing': compliance['missing_ppe']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def api_stats():
    """Get detection statistics"""
    if detector:
        return jsonify(detector.get_stats())
    return jsonify({'error': 'Detector not initialized'}), 500

# Main functions for command-line usage
def train_model(dataset_path, epochs=100):
    """Train a new PPE detection model"""
    
    print("Starting PPE model training...")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    
    trainer = PPETrainer(dataset_path)
    trainer.config.EPOCHS = epochs
    
    # Train the model
    results = trainer.train()
    
    # Evaluate performance
    metrics = trainer.evaluate()
    
    print("\nTraining completed!")
    return results, metrics

def test_detection(model_path, image_path):
    """Test detection on a single image"""
    
    print(f"Testing detection...")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    
    detector = PPEDetector(model_path)
    if not detector.model_loaded:
        print("Failed to load model")
        return
    
    # Load and process image
    image = Image.open(image_path)
    results = detector.detect(image)
    
    if results['success']:
        print(f"\nDetection Results:")
        print(f"Found {results['total_detections']} objects")
        print(f"Inference time: {results['inference_time_ms']:.1f}ms")
        
        print(f"\nCompliance: {results['compliance']['status']}")
        print(f"Score: {results['compliance']['score']:.2f}")
        
        if results['compliance']['missing_ppe']:
            print(f"Missing PPE: {', '.join(results['compliance']['missing_ppe'])}")
        
        print(f"\nDetected objects:")
        for detection in results['detections']:
            print(f"  {detection['class']}: {detection['confidence']:.3f}")
    else:
        print(f"Detection failed: {results['error']}")

def run_api(model_path, host='0.0.0.0', port=5000):
    """Start the Flask API server"""
    
    print(f"Starting PPE Detection API...")
    print(f"Model: {model_path}")
    
    if not init_api(model_path):
        print("Failed to initialize API - model loading failed")
        return
    
    print(f"API server starting on {host}:{port}")
    print(f"Endpoints:")
    print(f"  GET  /{host}:{port}/health")
    print(f"  POST /{host}:{port}/detect")
    print(f"  POST /{host}:{port}/detect_simple")
    print(f"  GET  /{host}:{port}/stats")
    
    app.run(host=host, port=port, debug=False)

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PPE Detection System')
    parser.add_argument('mode', choices=['train', 'test', 'api'], 
                       help='Operation mode')
    parser.add_argument('--dataset', help='Dataset path (for training)')
    parser.add_argument('--model', default='trained_models/best_ppe_model.pt',
                       help='Model path')
    parser.add_argument('--image', help='Image path (for testing)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--host', default='0.0.0.0', help='API host')
    parser.add_argument('--port', type=int, default=5000, help='API port')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.dataset:
            print("Error: --dataset required for training")
            sys.exit(1)
        train_model(args.dataset, args.epochs)
    
    elif args.mode == 'test':
        if not args.image:
            print("Error: --image required for testing")
            sys.exit(1)
        test_detection(args.model, args.image)
    
    elif args.mode == 'api':
        run_api(args.model, args.host, args.port)
  
