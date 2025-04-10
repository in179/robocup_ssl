#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>
RF24 radio(7, 8);
const byte address[6] = "00001";
int ans = 0;

void setup() {
  radio.begin();
  radio.openReadingPipe(1, address);
  radio.setPALevel(RF24_PA_MIN);
  Serial.begin(115200);
  Serial.setTimeout(2);
  //Serial.print(10);
}

void loop() {
  //delay(5);
  radio.startListening();
  if (radio.available()) {
    radio.read(&ans, sizeof(ans));
    Serial.print(ans);
    Serial.print("e");
    //delay(5);
  }
}
