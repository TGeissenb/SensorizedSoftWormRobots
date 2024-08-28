int raw1 = 0;
int raw2 = 0;
int raw3 = 0;
int raw4 = 0;
int raw5 = 0;
int raw6 = 0;
int raw7 = 0;
int raw8 = 0;
int raw9 = 0;
int raw10 = 0;
int raw11 = 0;
int raw12 = 0;
int raw13 = 0;
int raw14 = 0;
int raw15 = 0;
int raw16 = 0;
int Vin = 5;
float Vout = 0;
float R1 = 1000000;
float R2 = 0;
float buffer = 0;

void setup(){
Serial.begin(9600);
}

void loop(){
  raw1 = analogRead(A0);
  raw2 = analogRead(A1);
  raw3 = analogRead(A2);
  raw4 = analogRead(A3);
  raw5 = analogRead(A4);
  raw6 = analogRead(A5);
  raw7 = analogRead(A6);
  raw8 = analogRead(A7);
  raw9 = analogRead(A8);
  raw10 = analogRead(A9);
  raw11 = analogRead(A10);
  raw12 = analogRead(A11);
  raw13 = analogRead(A12);
  raw14 = analogRead(A13);
  raw15 = analogRead(A14);
  raw16 = analogRead(A15);

  Serial.print(millis());
  Serial.print(",");
  if(raw1){
    buffer = raw1 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw2){
    buffer = raw2 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw3){
    buffer = raw3 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw4){
    buffer = raw4 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw5){
    buffer = raw5 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw6){
    buffer = raw6 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw7){
    buffer = raw7 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw8){
    buffer = raw8 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw9){
    buffer = raw9 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw10){
    buffer = raw10 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw11){
    buffer = raw11 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw12){
    buffer = raw12 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw13){
    buffer = raw13 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw14){
    buffer = raw14 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw15){
    buffer = raw15 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.print(R2);
    Serial.print(",");
  }
  if(raw16){
    buffer = raw16 * Vin;
    Vout = (buffer)/1024.0;
    buffer = (Vin/Vout) - 1;
    R2= R1 * buffer;
    Serial.println(R2);
  }
  delay(100);
}