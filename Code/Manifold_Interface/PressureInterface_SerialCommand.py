import serial
import time

# Define a function to send a command via serial
def send_command(command):
    ser.write(command.encode())  # Encode the string and send it via serial
    print(f"Sent command: {command}")

# Define a function to receive data via serial
def receive_data():
    while True:
        if ser.in_waiting > 0:  # Check if there's data available to read
            received_data = ser.readline().decode().strip()  # Read the data and decode it
            print(f"Received data: {received_data}")


# Main program
if __name__ == '__main__':

    # Define the serial port settings
    port = 'COM9'  # Change this to the appropriate port on your system
    baud_rate = 9600  # Change this to match the baud rate of your device
    rec_freq = 30

    # Initialize the serial connection
    ser = serial.Serial(port, baud_rate, timeout=1)

    # Wait for the serial connection to establish
    time.sleep(2)

    """
    for i in range(50):
        print("Cycle {0:d}".format(i))
        send_command("SET;12;22;0;0;0;0;0;0;0\n")
        time.sleep(12)
        send_command("SET;3;0;0;0;0;0;0;0;0\n")
        time.sleep(3)
        send_command("SET;12;0;22;0;0;0;0;0;0\n")
        time.sleep(12)
        send_command("SET;3;0;0;0;0;0;0;0;0\n")
        time.sleep(3)
    """

    for _ in range(10):
        send_command("SET;2;10;0;0;0;0;0;0;0\n")
        time.sleep(2)
        send_command("SET;10;17;0;0;0;0;0;0;0\n")
        time.sleep(10)
        send_command("SET;2;17;0;0;0;0;0;0;0\n")
        time.sleep(2)
        send_command("SET;10;10;0;0;0;0;0;0;0\n")
        time.sleep(10)
        send_command("SET;2;0;0;0;0;0;0;0;0\n")
        time.sleep(2)
