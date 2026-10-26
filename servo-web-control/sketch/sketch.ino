// SPDX-License-Identifier: MPL-2.0
//
// Drives 12 MG90S servos on a PCA9685 (I2C) and reads an MPU6050 IMU on the
// same bus. Everything is exposed to the Python side (python/main.py) over the
// Bridge, which serves the web UI.
//
// Anything that touches I2C (servos, IMU init/calibrate/WHO_AM_I) is provide_safe:
// it runs in loop()'s thread, the same one that samples the IMU, so the two never
// use the bus at the same time. The two IMU READ handlers (getImu, imuBatch) only
// copy RAM, so they are plain `provide` (Bridge thread). That made NO measurable
// difference: every Bridge request still stalls the sampler for ~20 ms + ~2.4 ms
// per sample returned (the reply goes out over a slow UART), so polling
// imuBatch at ~25 Hz gives ~100 Hz with up to 210 ms gaps. Fine for a 20 Hz
// policy (one sample per tick); for the 200 Hz shake log, record on the MCU
// and download afterwards instead of streaming.

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Arduino_RouterBridge.h>
#include <math.h>

// (declared first: the Arduino build inserts function prototypes before the first function)
struct Sample { uint32_t t; float g[3]; float a[3]; };   // t in us; gyro rad/s; accel in g

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

// ------------------------------------------------------------------ IMU
// MPU6050 at 0x68 (AD0 low). Sampled at 200 Hz from loop(); everything is
// reported in the ROBOT frame: x toward L0 (front), y left, z up. If the chip
// is mounted turned, set the sensor->robot rotation with setImuAxes.
#define MPU_ADDR 0x68
#define IMU_PERIOD_US 5000          // 200 Hz
#define RING 256                    // 1.28 s of samples buffered for the host to fetch
#define DEG2RAD 0.017453292f

bool imuOk = false;
float gyroBias[3] = {0, 0, 0};                           // sensor frame, rad/s
float axes[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};   // robot = axes * sensor
float gEst[3] = {0, 0, -1};                              // down direction in the robot frame (unit)
Sample ring[RING];
volatile uint32_t head = 0;                              // monotonic counters, index = counter % RING;
uint32_t tail = 0;                                       // head: written by loop() only, tail: by the reader only
Sample latest;
uint32_t nextSampleUs = 0, lastUs = 0;

bool mpuRead(uint8_t reg, uint8_t* buf, uint8_t n) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

// raw sensor-frame reading: accel in g, gyro in rad/s (bias not removed)
bool readRaw(float a[3], float g[3]) {
  uint8_t b[14];
  if (!mpuRead(0x3B, b, 14)) return false;
  for (int i = 0; i < 3; i++) {
    a[i] = (int16_t)((b[2 * i] << 8) | b[2 * i + 1]) / 8192.0f;                    // +-4 g
    g[i] = (int16_t)((b[8 + 2 * i] << 8) | b[9 + 2 * i]) / 65.5f * DEG2RAD;        // +-500 dps
  }
  return true;
}

void toRobot(const float s[3], float r[3]) {
  for (int i = 0; i < 3; i++) r[i] = axes[i][0] * s[0] + axes[i][1] * s[1] + axes[i][2] * s[2];
}

bool imuInit() {
  uint8_t who = 0;
  if (!mpuRead(0x75, &who, 1) || who != 0x68) return false;      // WHO_AM_I
  mpuWrite(0x6B, 0x01);       // wake, clock = gyro X PLL
  delay(50);
  mpuWrite(0x1A, 0x03);       // low-pass filter ~44 Hz (accel) / 42 Hz (gyro)
  mpuWrite(0x19, 4);          // 1 kHz / (1 + 4) = 200 Hz internal rate
  mpuWrite(0x1B, 0x08);       // gyro +-500 deg/s
  mpuWrite(0x1C, 0x08);       // accel +-4 g
  delay(50);
  return true;
}

// Average the gyro for ~2 s with the robot STILL. Blocks for that long.
bool imuCalibrate() {
  if (!imuOk) return false;
  float sum[3] = {0, 0, 0}, a[3], g[3];
  int n = 0;
  for (int i = 0; i < 400; i++) {
    if (readRaw(a, g)) { for (int k = 0; k < 3; k++) sum[k] += g[k]; n++; }
    delay(5);
  }
  if (n < 300) return false;
  for (int k = 0; k < 3; k++) gyroBias[k] = sum[k] / n;
  return true;
}

void sampleImu() {
  float a[3], g[3], ar[3], gr[3];
  if (!readRaw(a, g)) return;
  for (int k = 0; k < 3; k++) g[k] -= gyroBias[k];
  toRobot(a, ar);
  toRobot(g, gr);
  uint32_t now = micros();
  float dt = lastUs ? (now - lastUs) * 1e-6f : IMU_PERIOD_US * 1e-6f;
  lastUs = now;
  if (dt > 0.05f) dt = IMU_PERIOD_US * 1e-6f;

  // gravity direction: integrate with the gyro (d g/dt = -w x g), pull toward the
  // accelerometer when it reads about 1 g (time constant ~0.25 s)
  float p[3] = {gEst[0] - (gr[1] * gEst[2] - gr[2] * gEst[1]) * dt,
                gEst[1] - (gr[2] * gEst[0] - gr[0] * gEst[2]) * dt,
                gEst[2] - (gr[0] * gEst[1] - gr[1] * gEst[0]) * dt};
  float an = sqrtf(ar[0] * ar[0] + ar[1] * ar[1] + ar[2] * ar[2]);
  float alpha = (fabsf(an - 1.0f) < 0.3f) ? 0.98f : 1.0f;
  float m[3] = {-ar[0] / an, -ar[1] / an, -ar[2] / an};          // accel measures "up"; down = -accel
  for (int k = 0; k < 3; k++) p[k] = alpha * p[k] + (1 - alpha) * m[k];
  float pn = sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  for (int k = 0; k < 3; k++) gEst[k] = p[k] / pn;

  Sample s;
  s.t = now;
  for (int k = 0; k < 3; k++) { s.g[k] = gr[k]; s.a[k] = ar[k]; }
  uint32_t h = head;                                             // (head++ below, after the slot is written)
  latest = s;
  ring[h % RING] = s;
  __asm volatile("" ::: "memory");                               // slot written before head moves
  head = h + 1;
}

int imuWho() {                      // WHO_AM_I register, or -1 if the chip doesn't answer
  uint8_t who = 0;
  return mpuRead(0x75, &who, 1) ? who : -1;
}

String fmt(const Sample& s) {
  String o = String(s.t);
  for (int k = 0; k < 3; k++) o += "," + String(s.g[k], 4);
  for (int k = 0; k < 3; k++) o += "," + String(s.a[k], 4);
  return o;
}

// latest sample plus the gravity estimate: "t_us,gx,gy,gz,ax,ay,az,grx,gry,grz"
String getImu() {
  if (!imuOk) return "";
  return fmt(latest) + "," + String(gEst[0], 4) + "," + String(gEst[1], 4) + "," + String(gEst[2], 4);
}

// up to maxN unread samples, oldest first, ';'-separated "t_us,gx,gy,gz,ax,ay,az"
String imuBatch(int maxN) {
  String o;
  if (!imuOk) return o;
  maxN = constrain(maxN, 1, 40);
  uint32_t h = head;
  if (h - tail > RING) tail = h - RING;                          // reader fell behind: skip to the oldest still intact
  while (tail < h && maxN-- > 0) {
    if (o.length()) o += ";";
    o += fmt(ring[tail % RING]);
    tail++;
  }
  return o;
}

// 9 numbers, row-major: robot = M * sensor
bool setImuAxes(String csv) {
  float v[9];
  int n = 0, start = 0;
  for (int i = 0; i <= (int)csv.length() && n < 9; i++) {
    if (i == (int)csv.length() || csv[i] == ',') { v[n++] = csv.substring(start, i).toFloat(); start = i + 1; }
  }
  if (n != 9) return false;
  for (int i = 0; i < 9; i++) axes[i / 3][i % 3] = v[i];
  gEst[0] = 0; gEst[1] = 0; gEst[2] = -1;
  return true;
}

void setup() {
  Bridge.begin();
  Monitor.begin(115200);

  pwm.begin();
  pwm.setPWMFreq(50);  // 50 Hz for standard servos
  Wire.setClock(400000);

  // Set all 12 servos to 90 degrees (middle position)
  for (int i = 0; i < NUM_SERVOS; i++) {
    setServoAngle(i, 90);
  }

  // IMU is optional: if it isn't there the servo control works as before.
  imuOk = imuInit();
  if (imuOk) {
    delay(1000);                    // let the servos settle, then take the gyro bias (robot must be still)
    imuCalibrate();
  }
  nextSampleUs = micros();

  // Expose these to the Linux-side Python app (see python/main.py)
  Bridge.provide_safe("setServoAngle", setServoAngle);
  Bridge.provide_safe("getServoAngle", getServoAngle);
  Bridge.provide_safe("setAll", setAll);
  Bridge.provide_safe("readServoRaw", readServoRaw);
  Bridge.provide_safe("imuWho", imuWho);
  Bridge.provide("getImu", getImu);          // RAM-only: served in the Bridge thread, never stalls loop()
  Bridge.provide("imuBatch", imuBatch);
  Bridge.provide_safe("imuCalibrate", imuCalibrate);
  Bridge.provide_safe("setImuAxes", setImuAxes);
}

void loop() {
  if (imuOk && (int32_t)(micros() - nextSampleUs) >= 0) {
    nextSampleUs += IMU_PERIOD_US;
    if ((int32_t)(micros() - nextSampleUs) > 0) nextSampleUs = micros() + IMU_PERIOD_US;   // fell behind: resync
    sampleImu();
  }
  delay(1);                         // yields; provide_safe handlers run between loop() calls
}
