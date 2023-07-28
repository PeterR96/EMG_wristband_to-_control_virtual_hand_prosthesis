#include <WiFi.h>
#include <WiFiClient.h>


#define sf 1000         // Change this for the desired sampling frequency
#define tc (1000 / sf)  // Time constant


const int numSensors = 6;
const int baselineSamples = 2000;
const int sensorPins[numSensors] = {34, 32, 39, 33, 36, 35 };
const int sampleRate = 1000;
unsigned long last_time = 0;

float readings[numSensors] = { 0 };
float baseline[numSensors] = { 0 };
boolean recording = false;

//server initialization
const char* ssid = "ESP32";
const char* password = "1";

WiFiServer server(80);
WiFiClient client;

void setup() {
  Serial.begin(500000);

  for (int i = 0; i < numSensors; i++) {
    pinMode(sensorPins[i], INPUT);
  }
  calculateBaseline();
  init_wifi();
  
}

void loop() {
  if (millis() - last_time >= tc) {
    last_time = millis();
    for (int i = 0; i < numSensors; i++) {
      readings[i] = (analogRead(sensorPins[i]) - baseline[i]) * 1000 * 3.3 / 4095.0;  // mV
      Serial.print(readings[i]);                                                   // Send the values with 2 decimal places
      if (i == numSensors - 1) {
        Serial.println();  // Print a semicolon after each sensor value
        //Serial.print(millis());
      } else {
        Serial.print(",");
      }
    }
  }
  delay(1);
}


void calculateBaseline() {
  Serial.println("Calculating baseline...");
  {
    if (millis() - last_time >= tc) {
      last_time = millis();
      for (int i = 0; i < numSensors; i++) {
        float sum = 0;
        for (int j = 0; j < baselineSamples; j++) {
          sum += analogRead(sensorPins[i]);
          delay(1);
        }
        baseline[i] = sum / baselineSamples;
        Serial.print(baseline[i]);  // Send the values with 2 decimal places
        if (i == numSensors - 1) {
          Serial.print(";");  // Print a semicolon after each baseline value
        } else {
          Serial.print(" - ");
        }
        delay(1);
      }
      Serial.println("Baseline calculated.");
    }
  }
}

void startRecording() {
  Serial.println("Starting recording...");
  recording = true;
}

void stopRecording() {
  Serial.println("Stopping recording.");
  recording = false;
}

void init_wifi() {
  // Configure the ESP32 as a Wi-Fi access point
  WiFi.softAP(ssid, password);
  IPAddress ip = WiFi.softAPIP();

  server.begin();
  Serial.println(ip);
  Serial.println("Server started");

}
