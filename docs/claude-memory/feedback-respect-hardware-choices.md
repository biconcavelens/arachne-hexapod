---
name: feedback-respect-hardware-choices
description: "The user's hardware decisions for Arachne are final; don't re-argue or upsell parts"
metadata:
  node_type: memory
  type: feedback
  originSessionId: cca4871e-0aac-46b0-8408-35683725756e
  modified: 2026-10-17T08:52:18.847Z
---

The user has made hardware choices and wants them respected, not re-argued: 2x MPU6050 IMU (use only the MPU6050 of the GY-87; I2C 0x68, same bus as the PCA9685 at 0x40); 2x 18650 in series (6.4-8.4 V, no regulator, even though it is above the MG90S 4.8-6 V rating); hot-glue-blob feet; declined an INA power monitor, ADS1115, BNO085 and spare servos for now.

**Why:** these were decided with cost and what is available to them in mind, and the fork that made them told me "respect them, don't re-argue". Reviewers' needs were already weighed.

**How to apply:** build around them. Where a choice has a side effect (7-8 V pack shifts servo behaviour with charge), mitigate in software or logging (log pack voltage every run, make voltage an input to the servo model) instead of asking for a part. The second robot is bought only after the kill test. Details: `research-questions.md`, "Hardware: the user's choices". Related: [[project-paper-a-servo-identification]].
