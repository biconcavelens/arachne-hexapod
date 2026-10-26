// SPDX-License-Identifier: MPL-2.0
//
// Drives 12 MG90S servos on a PCA9685 (I2C). setServoAngle/getServoAngle are
// exposed to the Python side (python/main.py) over the Bridge, which serves
// the web UI.

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Arduino_RouterBridge.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo pulse range for MG90S (adjust if needed)
#define SERVO_MIN 150   // ~0 degrees
#define SERVO_MAX 600   // ~180 degrees
#define NUM_SERVOS 12

int angles[NUM_SERVOS];

bool setServoAngle(int channel, int angle) {
  if (channel < 0 || channel >= NUM_SERVOS) return false;
  angle = constrain(angle, 0, 180);
  uint16_t pulse = map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
  pwm.setPWM(channel, 0, pulse);
  angles[channel] = angle;
  return true;
}

// All 12 channels in one call: "a0,a1,...,a11" (degrees). Used by the web
// page's animations so each frame is one Bridge message instead of twelve.
bool setAll(String csv) {
  int ch = 0, start = 0;
  for (int i = 0; i <= (int)csv.length() && ch < NUM_SERVOS; i++) {
    if (i == (int)csv.length() || csv[i] == ',') {
      setServoAngle(ch++, csv.substring(start, i).toInt());
      start = i + 1;
    }
  }
  return ch == NUM_SERVOS;
}

int getServoAngle(int channel) {
  if (channel < 0 || channel >= NUM_SERVOS) return -1;
  return angles[channel];
}

// Diagnostic only: reads back the "off" pulse count actually stored in the
// PCA9685's own registers for this channel, to tell an I2C/register problem
// apart from a downstream (wiring/power) one.
int readServoRaw(int channel) {
  if (channel < 0 || channel >= NUM_SERVOS) return -1;
  return pwm.getPWM(channel, true);
}

void setup() {
  Bridge.begin();
  Monitor.begin(115200);

  pwm.begin();
  pwm.setPWMFreq(50);  // 50 Hz for standard servos

  // Set all 12 servos to 90 degrees (middle position)
  for (int i = 0; i < NUM_SERVOS; i++) {
    setServoAngle(i, 90);
  }

  // Expose these to the Linux-side Python app (see python/main.py)
  Bridge.provide("setServoAngle", setServoAngle);
  Bridge.provide("getServoAngle", getServoAngle);
  Bridge.provide("setAll", setAll);
  Bridge.provide("readServoRaw", readServoRaw);
}

void loop() {
  delay(1000);
}
