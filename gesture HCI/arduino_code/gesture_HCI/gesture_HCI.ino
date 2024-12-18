#include <Servo.h>

Servo wrist;
Servo thumb;
Servo index;
Servo majeure;
Servo lingfinger;
Servo auriculaire;

String value;

void setup() {
  Serial.begin(9600);

  wrist.attach(3);

  thumb.attach(11);    // set servo pins
  index.attach(9);
  majeure.attach(6);
  lingfinger.attach(5);
  auriculaire.attach(10);
}

void loop() {
  if (Serial.available()) {
    value = Serial.readString(); // 입력된 값 전체 읽기

      // 문자열에서 각 서보 각도 파싱
    int wrist_angle = value.substring(0, 3).toInt();
    int thumb_angle = value.substring(3, 6).toInt();
    int index_angle = value.substring(6, 9).toInt();
    int majeure_angle = value.substring(9, 12).toInt();
    int lingfinger_angle = value.substring(12, 15).toInt();
    int auriculaire_angle = value.substring(15, 18).toInt();

    // 각 서보에 각도 전달
    wrist.write(wrist_angle);
    thumb.write(thumb_angle);
    index.write(index_angle);
    majeure.write(majeure_angle);
    lingfinger.write(lingfinger_angle);
    auriculaire.write(auriculaire_angle);
  }
}
