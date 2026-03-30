#define SENSOR_PIN A0   // ESP8266 ADC pin

// -------- PARAMETERS --------
#define CALIB_SAMPLES 100
#define WINDOW_SIZE 200
#define BUFFER_SIZE 200
#define DT 0.01
#define C_FLEX 40.0

// -------- LOGGING --------
#define PRINT_STATUS 1
#define PRINT_SAMPLE_STREAM 1   // real-time per-sample log
#define PRINT_FEATURES 1        // set to 1 to print full feature CSV rows
#define PRINT_RISK_PERCENT 1    // 0..100 risk score derived from LDH score

// -------- LDH HEURISTIC THRESHOLDS (TUNE THESE) --------
#define THRESH_RANGE_OF_MOTION 1.5
#define THRESH_MOTION_SPEED 25.0
#define THRESH_SMOOTHNESS 0.15
#define THRESH_JERK_RMS 8000.0
#define LDH_SCORE_MIN 2

// -------- BUFFERS --------
float dataWindow[WINDOW_SIZE];
float zBuffer[BUFFER_SIZE];
int sampleIndex = 0;
int bufferIndex = 0;
int straightCountWindow = 0;

// -------- VARIABLES --------
float offset = 0;
float z = 0;
String currentState = "STRAIGHT";
const char* lastLdhFlag = "NO_LDH";
int lastLdhScore = 0;

// -------- FEATURE EXTRACTION FUNCTION --------
void computeFeatures() {
  float sumV = 0;
  float sqSumV = 0;
  float maxV = -100000;
  float minV = 100000;

  // velocity and jerk stats
  float sumVel = 0;
  float sumAbsVel = 0;
  float maxAbsVel = 0;
  float sumVelSq = 0;

  float sumJerkSq = 0;

  for (int i = 0; i < WINDOW_SIZE; i++) {
    float v = dataWindow[i];

    sumV += v;
    sqSumV += v * v;
    if (v > maxV) maxV = v;
    if (v < minV) minV = v;

    if (i > 0) {
      float vel = (dataWindow[i] - dataWindow[i - 1]) / DT;
      sumVel += vel;
      sumAbsVel += fabs(vel);
      sumVelSq += vel * vel;
      if (fabs(vel) > maxAbsVel) maxAbsVel = fabs(vel);

      if (i > 1) {
        float prevVel = (dataWindow[i - 1] - dataWindow[i - 2]) / DT;
        float jerk = (vel - prevVel) / DT;
        sumJerkSq += jerk * jerk;
      }
    }
  }

  float meanV = sumV / WINDOW_SIZE;
  float rangeOfMotion = maxV - minV;

  float meanVel = sumVel / (WINDOW_SIZE - 1);
  float motionSpeed = sumAbsVel / (WINDOW_SIZE - 1);
  float maxVelocity = maxAbsVel;

  // std of velocity
  float meanVelSq = sumVelSq / (WINDOW_SIZE - 1);
  float velStd = sqrt(fmaxf(0.0f, meanVelSq - meanVel * meanVel));

  float smoothness = (velStd == 0) ? 0 : (meanV / velStd);

  float jerkRMS = sqrt(sumJerkSq / max(1, WINDOW_SIZE - 2));

  float postureDuration = (float)straightCountWindow / WINDOW_SIZE;

  // -------- HEURISTIC LDH FLAG (NOT A DIAGNOSIS) --------
  int ldhScore = 0;
  if (rangeOfMotion > THRESH_RANGE_OF_MOTION) ldhScore++;
  if (motionSpeed > THRESH_MOTION_SPEED) ldhScore++;
  if (smoothness < THRESH_SMOOTHNESS) ldhScore++;
  if (jerkRMS > THRESH_JERK_RMS) ldhScore++;

  const char* ldhFlag = (ldhScore >= LDH_SCORE_MIN) ? "LDH" : "NO_LDH";

  lastLdhFlag = ldhFlag;
  lastLdhScore = ldhScore;

#if PRINT_RISK_PERCENT
  float riskPercent = (ldhScore / 4.0f) * 100.0f;
#endif

#if PRINT_FEATURES
  // -------- PRINT FEATURE CSV ROW --------
  Serial.print("W,"); Serial.print(millis()); Serial.print(",");
  Serial.print(rangeOfMotion); Serial.print(",");
  Serial.print(motionSpeed); Serial.print(",");
  Serial.print(postureDuration); Serial.print(",");
  Serial.print(smoothness); Serial.print(",");
  Serial.print(jerkRMS); Serial.print(",");
  Serial.print(meanVel); Serial.print(",");
  Serial.print(maxVelocity); Serial.print(",");
  Serial.print(currentState); Serial.print(",");
  Serial.print(ldhFlag); Serial.print(",");
  Serial.print(ldhScore);
#if PRINT_RISK_PERCENT
  Serial.print(",");
  Serial.println(riskPercent, 1);
#else
  Serial.println();
#endif
#else
  // -------- PRINT LDH PREDICTION ONLY --------
  Serial.print("W,"); Serial.print(millis()); Serial.print(",");
  Serial.print(ldhFlag); Serial.print(",");
#if PRINT_RISK_PERCENT
  Serial.print(ldhScore); Serial.print(",");
  Serial.println(riskPercent, 1);
#else
  Serial.println(ldhScore);
#endif
#endif
}

// -------- SETUP --------
void setup() {
  Serial.begin(115200);
  delay(1000);

#if PRINT_SAMPLE_STREAM
  Serial.println("S,ms,raw,v,D,state");
#endif
#if PRINT_FEATURES
  Serial.println("W,ms,RangeOfMotion,MotionSpeed,PostureDuration,Smoothness,JerkRMS,MeanVelocity,MaxVelocity,Label,LDH,Score,RiskPercent");
#else
#if PRINT_RISK_PERCENT
  Serial.println("W,ms,LDH,Score,RiskPercent");
#else
  Serial.println("W,ms,LDH,Score");
#endif
#endif

  // ---- OFFSET CALIBRATION ----
#if PRINT_STATUS
  Serial.println("# Calibrating... Keep sensor STRAIGHT");
#endif

  for (int i = 0; i < CALIB_SAMPLES; i++) {
    offset += analogRead(SENSOR_PIN);
    delay(10);
  }

  offset /= CALIB_SAMPLES;

#if PRINT_STATUS
  Serial.println("# Calibration done");
#endif
}

// -------- LOOP --------
void loop() {
  int raw = analogRead(SENSOR_PIN);

  // Offset removal (voltage only)
  float v = raw - offset;

  // Integration for posture detection (same as original behavior)
  z = z + v * DT;
  zBuffer[bufferIndex] = z;
  int oldIndex = (bufferIndex + 1) % BUFFER_SIZE;
  float z_old = zBuffer[oldIndex];
  float D = z - z_old;
  bufferIndex = oldIndex;

  // -------- POSTURE DETECTION --------
  if (D > C_FLEX)
    currentState = "BENDING";
  else if (D < -C_FLEX)
    currentState = "RETURNING";
  else
    currentState = "STRAIGHT";

#if PRINT_SAMPLE_STREAM
  // -------- REAL-TIME SAMPLE LOG --------
  Serial.print("S,"); Serial.print(millis()); Serial.print(",");
  Serial.print(raw); Serial.print(",");
  Serial.print(v); Serial.print(",");
  Serial.print(D); Serial.print(",");
  Serial.println(currentState);
#endif

  // Store sample
  dataWindow[sampleIndex] = v;
  if (currentState == "STRAIGHT") {
    straightCountWindow++;
  }

  sampleIndex++;

  // -------- WINDOW COMPLETE --------
  if (sampleIndex >= WINDOW_SIZE) {
    computeFeatures();
    sampleIndex = 0;
    straightCountWindow = 0;
  }

  delay(10);
}
