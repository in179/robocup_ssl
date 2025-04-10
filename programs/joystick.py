import time

import serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
for port in ports:
    print(f"Порт: {port.device}")
    print(f"Описание: {port.description}")
    print(f"Производитель: {port.manufacturer}\n")

import serial
#{input("Номер порта: ")}
ser = serial.Serial(f'/dev/cu.usbserial-A5069RR4', baudrate=115200, timeout=0.001)
time.sleep(2)

def send(x):
    ser.write(x.encode())
    time.sleep(0.01)

def motors(a, b, c):
    a, b, c = int(a), int(b), int(c)
    a = max(-100, min(100, a))
    b = max(-100, min(100, b))
    c = max(-100, min(100, c))
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

#motors = fake

import pygame, math
pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Joystick Control")
clock = pygame.time.Clock()
center = (width//2, height//2)
max_radius = 100

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    mouse_pos = pygame.mouse.get_pos()
    jx = mouse_pos[0] - center[0]
    jy = center[1] - mouse_pos[1]
    mag = math.hypot(jx, jy)
    if mag > max_radius:
        scale = max_radius / mag
        jx *= scale
        jy *= scale
        clamped_pos = (center[0] + jx, center[1] - jy)
    else:
        clamped_pos = mouse_pos

    vx = (jx / max_radius) * 100
    vy = (jy / max_radius) * 100
    motor_a = vx + vy
    motor_b = -vx
    motor_c = -vy
    motors(motor_a, motor_b, motor_c)

    screen.fill((30, 30, 30))
    pygame.draw.circle(screen, (80, 80, 80), center, max_radius, 2)
    pygame.draw.circle(screen, (255, 0, 0), center, 5)
    pygame.draw.line(screen, (0, 255, 0), center, clamped_pos, 3)
    pygame.draw.circle(screen, (0, 0, 255), (int(clamped_pos[0]), int(clamped_pos[1])), 8)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()
