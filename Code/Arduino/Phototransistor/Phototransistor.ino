int resist_sum_1 = 0;
int resist_sum_2 = 0;
int resist_sum_3 = 0;
int resist_sum_4 = 0;
int resist_sum_5 = 0;
int resist_sum_6 = 0;

int sampling = 100;

void setup() {
  // put your setup code here, to run once:
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  pinMode(A4, INPUT);
  pinMode(A5, INPUT);
  pinMode(A6, INPUT);
  pinMode(A7, INPUT);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  // int photo_resist_1 = analogRead(A5);
  // int photo_resist_2 = analogRead(A6);

  for(int i=0;i<sampling;i++){
    int photo_resist_1 = analogRead(A2);
    int photo_resist_2 = analogRead(A3);
    int photo_resist_3 = analogRead(A4);
    int photo_resist_4 = analogRead(A5);
    int photo_resist_5 = analogRead(A6);
    int photo_resist_6 = analogRead(A7);
    resist_sum_1 += photo_resist_1;
    resist_sum_2 += photo_resist_2;
    resist_sum_3 += photo_resist_3;
    resist_sum_4 += photo_resist_4;
    resist_sum_5 += photo_resist_5;
    resist_sum_6 += photo_resist_6;
  }

  Serial.print(millis());
  Serial.print(",");
  Serial.print(resist_sum_1);
  Serial.print(",");
  Serial.print(resist_sum_2);
  Serial.print(",");
  Serial.print(resist_sum_3);
  Serial.print(",");
  Serial.print(resist_sum_4);
  Serial.print(",");
  Serial.print(resist_sum_5);
  Serial.print(",");
  Serial.println(resist_sum_6);
  resist_sum_1 = 0;
  resist_sum_2 = 0;
  resist_sum_3 = 0;
  resist_sum_4 = 0;
  resist_sum_5 = 0;
  resist_sum_6 = 0;
}
