/*
 * ESP32 Safety Monitoring System with OV7670 Camera
 * Multi-sensor monitoring with Wi-Fi transmission
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <esp_camera.h>
#include <esp_sleep.h>
#include <ArduinoJson.h>
#include <img_converters.h>

// Wi-Fi credentials
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL = "http://your-server.com/api/upload";

// Pin definitions
#define DHT_PIN 14
#define GAS_SENSOR_PIN 36
#define MOISTURE_PIN 39
#define ULTRASONIC_TRIG 32
#define ULTRASONIC_ECHO 33
#define LED_RED 2
#define LED_YELLOW 15
#define BUZZER_PIN 12

// Thresholds
#define GAS_THRESHOLD 500
#define TEMP_THRESHOLD 40.0
#define MOISTURE_THRESHOLD 300
#define PROXIMITY_THRESHOLD 100

// Timing
#define SENSOR_READ_INTERVAL 5000
#define CAMERA_INTERVAL 30000
#define DEEP_SLEEP_DURATION 60

DHT dht(DHT_PIN, DHT22);
bool alertActive = false;
unsigned long lastSensorRead = 0;
unsigned long lastCameraCapture = 0;

struct SensorData {
  float temperature;
  float humidity;
  int gasLevel;
  int moistureLevel;
  int proximityDistance;
  bool personNearby;
  bool alertTriggered;
};

// OV7670 camera configuration
camera_config_t camera_config = {
  .pin_pwdn = -1,
  .pin_reset = 13,
  .pin_xclk = 21,
  .pin_sscb_sda = 26,
  .pin_sscb_scl = 27,
  .pin_d7 = 35,
  .pin_d6 = 34,
  .pin_d5 = 39,
  .pin_d4 = 36,
  .pin_d3 = 19,
  .pin_d2 = 18,
  .pin_d1 = 5,
  .pin_d0 = 4,
  .pin_vsync = 25,
  .pin_href = 23,
  .pin_pclk = 22,
  .xclk_freq_hz = 10000000,
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,
  .pixel_format = PIXFORMAT_RGB565,
  .frame_size = FRAMESIZE_QVGA,
  .jpeg_quality = 10,
  .fb_count = 2,
  .grab_mode = CAMERA_GRAB_WHEN_EMPTY
};

void initializeSystem() {
  Serial.begin(115200);
  Serial.println("Starting ESP32 Safety Monitor...");
  
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(ULTRASONIC_TRIG, OUTPUT);
  pinMode(ULTRASONIC_ECHO, INPUT);
  
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  
  dht.begin();
  Serial.println("System initialized");
}

bool initializeCamera() {
  esp_err_t err = esp_camera_init(&camera_config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  
  sensor_t * s = esp_camera_sensor_get();
  if (s == NULL) {
    Serial.println("Failed to get camera sensor");
    return false;
  }
  
  s->set_brightness(s, 2);
  s->set_contrast(s, 1);
  s->set_saturation(s, 0);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_exposure_ctrl(s, 1);
  s->set_aec2(s, 0);
  s->set_ae_level(s, 0);
  s->set_aec_value(s, 300);
  s->set_gain_ctrl(s, 1);
  s->set_gainceiling(s, GAINCEILING_2X);
  
  Serial.println("OV7670 camera initialized");
  delay(1000);
  return true;
}

bool connectToWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.printf("WiFi connected: %s\n", WiFi.localIP().toString().c_str());
    return true;
  }
  
  Serial.println("\nWiFi connection failed");
  return false;
}

void disconnectWiFi() {
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
}

SensorData readAllSensors() {
  SensorData data;
  
  data.temperature = dht.readTemperature();
  data.humidity = dht.readHumidity();
  
  if (isnan(data.temperature) || isnan(data.humidity)) {
    data.temperature = 0;
    data.humidity = 0;
  }
  
  data.gasLevel = analogRead(GAS_SENSOR_PIN);
  data.moistureLevel = analogRead(MOISTURE_PIN);
  data.proximityDistance = readUltrasonicDistance();
  data.personNearby = (data.proximityDistance > 0 && data.proximityDistance < PROXIMITY_THRESHOLD);
  
  return data;
}

int readUltrasonicDistance() {
  digitalWrite(ULTRASONIC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG, LOW);
  
  long duration = pulseIn(ULTRASONIC_ECHO, HIGH, 30000);
  if (duration == 0) return -1;
  
  return duration * 0.034 / 2;
}

bool captureAndSendImage() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return false;
  }
  
  Serial.printf("Image captured: %d bytes\n", fb->len);
  bool success = convertAndSendImage(fb);
  esp_camera_fb_return(fb);
  
  return success;
}

bool convertAndSendImage(camera_fb_t* fb) {
  if (fb->format != PIXFORMAT_RGB565) {
    Serial.println("Unexpected pixel format");
    return false;
  }
  
  size_t jpg_buf_len = 0;
  uint8_t * jpg_buf = NULL;
  
  bool converted = fmt2jpg(fb->buf, fb->len, fb->width, fb->height, 
                          PIXFORMAT_RGB565, 80, &jpg_buf, &jpg_buf_len);
  
  if (!converted) {
    Serial.println("JPEG conversion failed, sending raw data");
    return sendRawImageToServer(fb->buf, fb->len);
  }
  
  bool success = sendImageToServer(jpg_buf, jpg_buf_len);
  
  if (jpg_buf) {
    free(jpg_buf);
  }
  
  return success;
}

bool sendImageToServer(uint8_t* imageData, size_t imageSize) {
  if (WiFi.status() != WL_CONNECTED) return false;
  
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("Content-Length", String(imageSize));
  
  int httpResponseCode = http.POST(imageData, imageSize);
  http.end();
  
  if (httpResponseCode > 0) {
    Serial.printf("Image upload: %d\n", httpResponseCode);
    return httpResponseCode == 200;
  }
  
  Serial.printf("Upload failed: %s\n", http.errorToString(httpResponseCode).c_str());
  return false;
}

bool sendRawImageToServer(uint8_t* imageData, size_t imageSize) {
  if (WiFi.status() != WL_CONNECTED) return false;
  
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("X-Image-Format", "RGB565");
  http.addHeader("X-Image-Width", "320");
  http.addHeader("X-Image-Height", "240");
  
  int httpResponseCode = http.POST(imageData, imageSize);
  http.end();
  
  return httpResponseCode == 200;
}

bool evaluateSafetyConditions(const SensorData& data) {
  bool gasAlert = (data.gasLevel > GAS_THRESHOLD);
  bool tempAlert = (data.temperature > TEMP_THRESHOLD);
  bool moistureAlert = (data.moistureLevel > MOISTURE_THRESHOLD);
  
  bool criticalCondition = (gasAlert || tempAlert || moistureAlert) && data.personNearby;
  
  if (criticalCondition) {
    Serial.println("⚠️  SAFETY ALERT!");
    Serial.printf("Gas:%d Temp:%.1f°C Moisture:%d Person:%s\n",
                  data.gasLevel, data.temperature, data.moistureLevel, 
                  data.personNearby ? "YES" : "NO");
  }
  
  return criticalCondition;
}

void triggerAlert(bool active) {
  if (active) {
    digitalWrite(LED_RED, HIGH);
    digitalWrite(LED_YELLOW, LOW);
    
    for (int i = 0; i < 3; i++) {
      digitalWrite(BUZZER_PIN, HIGH);
      delay(200);
      digitalWrite(BUZZER_PIN, LOW);
      delay(200);
    }
  } else {
    digitalWrite(LED_RED, LOW);
    digitalWrite(LED_YELLOW, HIGH);
    digitalWrite(BUZZER_PIN, LOW);
    delay(100);
    digitalWrite(LED_YELLOW, LOW);
  }
  
  alertActive = active;
}

void sendSensorData(const SensorData& data) {
  if (WiFi.status() != WL_CONNECTED) return;
  
  DynamicJsonDocument json(512);
  json["timestamp"] = millis();
  json["temperature"] = data.temperature;
  json["humidity"] = data.humidity;
  json["gas_level"] = data.gasLevel;
  json["moisture"] = data.moistureLevel;
  json["proximity"] = data.proximityDistance;
  json["person_nearby"] = data.personNearby;
  json["alert_active"] = data.alertTriggered;
  
  String jsonString;
  serializeJson(json, jsonString);
  
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  
  int httpResponseCode = http.POST(jsonString);
  http.end();
  
  if (httpResponseCode > 0) {
    Serial.printf("Data sent: %d\n", httpResponseCode);
  } else {
    Serial.println("Data transmission failed");
  }
}

void enterDeepSleep() {
  Serial.println("Entering deep sleep...");
  disconnectWiFi();
  esp_camera_deinit();
  esp_sleep_enable_timer_wakeup(DEEP_SLEEP_DURATION * 1000000ULL);
  esp_deep_sleep_start();
}

void optimizePowerConsumption() {
  setCpuFrequencyMhz(80);
  analogSetAttenuation(ADC_11db);
}

void setup() {
  initializeSystem();
  optimizePowerConsumption();
  
  if (!initializeCamera()) {
    Serial.println("Camera failed. Continuing without camera...");
  }
  
  if (!connectToWiFi()) {
    Serial.println("WiFi failed. Operating offline...");
  }
  
  Serial.println("System ready");
}

void loop() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastSensorRead >= SENSOR_READ_INTERVAL) {
    SensorData sensorData = readAllSensors();
    sensorData.alertTriggered = evaluateSafetyConditions(sensorData);
    triggerAlert(sensorData.alertTriggered);
    sendSensorData(sensorData);
    
    Serial.printf("T:%.1f°C H:%.1f%% G:%d M:%d D:%dcm Alert:%s\n",
                  sensorData.temperature, sensorData.humidity, sensorData.gasLevel,
                  sensorData.moistureLevel, sensorData.proximityDistance,
                  sensorData.alertTriggered ? "YES" : "NO");
    
    lastSensorRead = currentTime;
  }
  
  if (currentTime - lastCameraCapture >= CAMERA_INTERVAL) {
    if (WiFi.status() == WL_CONNECTED) {
      captureAndSendImage();
    }
    lastCameraCapture = currentTime;
  }
  
  delay(100);
}
