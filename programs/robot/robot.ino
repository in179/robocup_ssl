int fin1 = 3;
int fin2 = 5;
int sin1 = 6;
int sin2 = 9;
int tin1 = 10;
int tin2 = 11;

void setup() {
  pinMode(fin1, OUTPUT);
  pinMode(fin2, OUTPUT);
  pinMode(sin1, OUTPUT);
  pinMode(sin2, OUTPUT);
  pinMode(tin1, OUTPUT);
  pinMode(tin2, OUTPUT);
  Serial.begin(115200);
  Serial.print(10);
  Serial.setTimeout(2);
}

void motor(int in1, int in2, int s) {
  s = map(s, -100, 100, -255, 255);
  if (s >= 0) {
    analogWrite(in1, 0);
    analogWrite(in2, s);
  } else {
    s = abs(s);
    analogWrite(in1, s);
    analogWrite(in2, 0);
  }
}

void motors(int a, int b, int c) {
  motor(fin1, fin2, a);
  motor(sin1, sin2, b);
  motor(tin1, tin2, c);
}

int parse() {
  bool flag = false;
  String now = "";
  while (true) {
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
      int ans = now.toInt();
      return ans;
    }
  }
}

void loop() {
  int z = parse();
  z -= 100;
  Serial.print("z = ");
  Serial.println(z);
  if (z == -1) {
    int a = parse() - 100;
    int b = parse() - 100;
    int c = parse() - 100;
    Serial.println(a);
    Serial.println(b);
    Serial.println(c);
    motors(a, b, c);
  }
  //motors(100, 100, 0);
  /*
  for (int i = 100; i >= -100; i--) {
    motors(0, i, 0);
    delay(25);
  }
  for (int i = -100; i <= 100; i++) {
    motors(0, i, 0);
    delay(25);
  }*/
}
