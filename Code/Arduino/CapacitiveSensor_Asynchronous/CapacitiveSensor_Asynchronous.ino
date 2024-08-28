#include <CapacitiveSensor.h>

int threshold = 30;

CapacitiveSensor cs1 = CapacitiveSensor(10, 11);
CapacitiveSensor cs2 = CapacitiveSensor(8, 9);
CapacitiveSensor cs3 = CapacitiveSensor(6, 7);
CapacitiveSensor cs4 = CapacitiveSensor(4, 5);
CapacitiveSensor cs5 = CapacitiveSensor(2, 3);

float total1 =  0;
float total2 =  0;
float total3 =  0;
float total4 =  0;
float total5 =  0;

int sensor_id = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  sensor_id ++;
  switch(sensor_id){
    case 2:
      total1 =  cs1.capacitiveSensorRaw(100);
      break;
    case 4:
      total2 =  cs2.capacitiveSensorRaw(100);
      break;
    case 6:
      total3 =  cs3.capacitiveSensorRaw(100);
      break;
    case 8:
      total4 =  cs4.capacitiveSensorRaw(100);
      break;
    case 10:
      total5 =  cs5.capacitiveSensorRaw(100);
      sensor_id = 0;
      break;
    default:
      delay(10);
  }

  Serial.print(millis());
  Serial.print(",");
  Serial.print(total1);
  Serial.print(",");
  Serial.print(total2);
  Serial.print(",");
  Serial.print(total3);
  Serial.print(",");
  Serial.print(total4);
  //Serial.print(",");
  //Serial.print(total5);
  Serial.println();

}