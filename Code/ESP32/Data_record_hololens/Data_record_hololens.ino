#include <WiFi.h>
#include <WiFiClient.h>

#define sf 1000           //change this for wanted sampling fq
#define tc (1000 / (sf))  // time constant

const int numSensors = 6;
const int baselineSamples = 1000;
const int sensorPins[numSensors] = {  34, 32, 39, 33, 36, 35 };
const int sampleRate = 1000;
unsigned long last_time = 0;
boolean recording = false;
boolean init_flag = false;

float baseline[numSensors] = { 0 };
float readings[numSensors] = { 0 };

const char* ssid = "ESP32";
const char* password = "1";


WiFiServer server(80);
WiFiClient client;

void setup() {
  Serial.begin(500000);

  for (int i = 0; i < numSensors; i++) {
    pinMode(sensorPins[i], INPUT);
  }
}

void loop() {

  if (!init_flag) {
    char command = Serial.read();
    if (command == 'c') {
      calculateBaseline();
      init_flag = true;
      init_wifi();
    }
  }
  if (init_flag) {
    client = server.available();
    if (client) {
      Serial.println("Client connected");

      while (client.connected()) {
        if (client.available()) {

          String request = client.readStringUntil('\r');
          Serial.print("Received request: ");
          Serial.println(request);
          server.close();
          client.stop();
          //WiFi.softAPdisconnect();
          //WiFi.disconnect();  // Disconnect from WiFi
          //WiFi.mode(WIFI_OFF); 
          init_flag = false;
        }
      }
      startRecording();
    }
  }
  
  if (recording) {
    if (millis() - last_time >= tc) {
      last_time = millis();
      for (int i = 0; i < numSensors; i++) {
        readings[i] = (analogRead(sensorPins[i]) - baseline[i]) * 1000 * 3.3 / 4095.0;  //mV
        Serial.print(readings[i]);
        if (i == 5) {
          Serial.println("");
        } else {
          Serial.print("; ");
        }
      }
    }
  }
}

void calculateBaseline() {
  Serial.println("Calculating baseline...");

  //if (millis() - last_time >= tc) {
  last_time = millis();
  for (int i = 0; i < numSensors; i++) {
    float sum = 0;
    for (int j = 0; j < baselineSamples; j++) {
      sum += analogRead(sensorPins[i]);
      delay(1);
    }
    baseline[i] = sum / baselineSamples;
    Serial.print(baseline[i]);
    Serial.print("; ");
    delay(1);
  }
  Serial.println("Baseline calculated.");
  //}
}

void startRecording() {
  recording = true;
}

void init_wifi() {
  // Configure the ESP32 as a Wi-Fi access point
  WiFi.softAP(ssid, password);
  IPAddress ip = WiFi.softAPIP();
  server.begin();
  Serial.println("Server started");
}

void setup_serial(){
  Serial.begin(500000);

  for (int i = 0; i < numSensors; i++) {
    pinMode(sensorPins[i], INPUT);
  }
}
