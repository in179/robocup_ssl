#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>
RF24 radio(7, 8);
const byte address[6] = "00001";
int ans = 0;

void setup() {
  radio.begin();
  radio.openWritingPipe(address);
  radio.setPALevel(RF24_PA_MIN);
  radio.stopListening();
  Serial.begin(115200);
  Serial.setTimeout(2);
}
String now = "";
bool flag = false;

void loop() {
  if (Serial.available()) {
    while (Serial.available()) {
      char x = Serial.read();
      if (x == 'e') {
        flag = true;
        break;
      }
      if (x == '-' || (x >= '0' && x <= '9')) {
        now += x;
      }
    }
    //Serial.println(now);
    if (flag) {
      //Serial.println(now);
      ans = now.toInt();
      radio.write(&ans, sizeof(ans));
      //delay(1000);
      Serial.println(ans);
      flag = false;
      now = "";
      //Serial.println();
    }
  }
}
