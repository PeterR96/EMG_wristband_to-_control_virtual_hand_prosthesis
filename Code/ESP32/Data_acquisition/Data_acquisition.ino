#define sf 1000           //change this for wanted sampling fq
#define tc (1000 / (sf))  // time constant

const int numSensors = 6;
const int baselineSamples = 2000;
const int sensorPins[numSensors] = { 34, 32, 39, 33, 36, 35 };
const int sampleRate = 1000;
unsigned long last_time = 0;

float baseline[numSensors] = { 0 };
float readings[numSensors] = { 0 };

const float x = 4096.0 * 3.3;
boolean recording = false;

void setup() {
  Serial.begin(500000);

  for (int i = 0; i < numSensors; i++) {
    pinMode(sensorPins[i], INPUT);
  }
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'c') {
      calculateBaseline();
    } else if (command == 'r') {
      startRecording();
    } else if (command == 'e') {
      stopRecording();
    }
  }

  if (recording) {
    if (millis() - last_time >= tc) {
      last_time = millis();
      for (int i = 0; i < numSensors; i++) {
        readings[i] = (analogRead(sensorPins[i]) - baseline[i])*1000*3.3/4095.0 ;  //mV
        Serial.print(readings[i]);
        if (i == 5) {
          Serial.print("");
        } else {
          Serial.print("; ");
        }
      }
      Serial.println();
    }
  }
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
        Serial.print(baseline[i]);
        Serial.print("; ");
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

