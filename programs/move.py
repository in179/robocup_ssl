import time

import serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
for port in ports:
    print(f"Порт: {port.device}")
    print(f"Описание: {port.description}")
    print(f"Производитель: {port.manufacturer}\n")

import serial
#{input("Номер порта: ")}
#ser = serial.Serial(f'/dev/cu.usbserial-A5069RR4', baudrate=115200, timeout=0.001)
time.sleep(2)

def send(x):
    ser.write(x.encode())
    time.sleep(0.01)

def motors(a, b, c):
    a, b, c = int(a), int(b), int(c)
    a = max(-100, min(100, a))
    b = max(-100, min(100, b))
    v = max(-100, min(100, c))
    a += 100
    b += 100
    c += 100
    #print(f"({a}, {b}, {c})", end="")
    send(f"99e{a}e{b}e{c}e")
    #time.sleep(0.3)
    '''
    while ser.in_waiting:
        print(ser.read(ser.in_waiting).decode(), end="")'''

def fake(*args, **kwargs):
    pass

motors = fake

if __name__ == "__main__":
    base = 80
    motors(-base, base, base)
    time.sleep(2)
    motors(0, 0, 0)
    time.sleep(2)
    while ser.in_waiting:
        print(ser.read(1).decode(), end="")