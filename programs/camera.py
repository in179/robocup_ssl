import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras
from PIL import Image
from move import motors
import time
import math
from math import sqrt

np.seterr(all='warn')

for camera_info in enumerate_cameras():
    print(f'{camera_info.index}: {camera_info.name}')
index = 1201 #int(input("Camera index is "))
cap = cv2.VideoCapture(index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
w = cap.get(3)
h = cap.get(4)
M = (-1, -1)
print(w, h)
img = 0
bpovmin, bpovmax = 70, 80
bmovemin, bmovemax = 75, 90
phase = 0

def photo():
    global img
    ret, img = cap.read()

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = -y
    
    def angle(self):
        return math.degrees(math.atan2(self.y, self.x))
    
    def len(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

class Line:
    def __init__(self, A, B):
        self.a = A[1] - B[1] + 0.0000000000000001
        self.b = B[0] - A[0]
        self.c = - (self.a * A[0] + self.b* A[1])
    
    def intersect(self, d, e, r):
        a, b, c = self.a, self.b, self.c
        x1=(-b*(-(a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(-2*a*c-2*e*a*b)*d-c**2-2*e*b*c-e**2*b**2)))/(b**2+a**2)-(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)+(e*a**2)/(b**2+a**2))-c)/a
        y1=-(a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(-2*a*c-2*e*a*b)*d-c**2-2*e*b*c-e**2*b**2)))/(b**2+a**2)-(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)+(e*a**2)/(b**2+a**2)
        x2=(-b*((a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(-2*a*c-2*e*a*b)*d-c**2-2*e*b*c-e**2*b**2)))/(b**2+a**2)-(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)+(e*a**2)/(b**2+a**2))-c)/a
        y2=(a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(-2*a*c-2*e*a*b)*d-c**2-2*e*b*c-e**2*b**2)))/(b**2+a**2)-(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)+(e*a**2)/(b**2+a**2)
        '''
        x3=(-b*(-(a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(2*a*c-2*e*a*b)*d-c**2+2*e*b*c-e**2*b**2)))/(b**2+a**2)+(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)-(e*a**2)/(b**2+a**2))-c)/a
        y3=-(a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(2*a*c-2*e*a*b)*d-c**2+2*e*b*c-e**2*b**2)))/(b**2+a**2)+(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)-(e*a**2)/(b**2+a**2)
        x4=(-b*((a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(2*a*c-2*e*a*b)*d-c**2+2*e*b*c-e**2*b**2)))/(b**2+a**2)+(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)-(e*a**2)/(b**2+a**2))-c)/a
        y4=(a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(2*a*c-2*e*a*b)*d-c**2+2*e*b*c-e**2*b**2)))/(b**2+a**2)+(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)-(e*a**2)/(b**2+a**2)
        x5=(-b*(-(a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(2*e*a*b-2*a*c)*d-c**2+2*e*b*c-e**2*b**2)))/(b**2+a**2)-(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)-(e*a**2)/(b**2+a**2))-c)/a
        y5=-(a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(2*e*a*b-2*a*c)*d-c**2+2*e*b*c-e**2*b**2)))/(b**2+a**2)-(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)-(e*a**2)/(b**2+a**2)
        x6=(-b*((a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(2*e*a*b-2*a*c)*d-c**2+2*e*b*c-e**2*b**2)))/(b**2+a**2)-(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)-(e*a**2)/(b**2+a**2))-c)/a
        y6=(a*sqrt(abs((b**2+a**2)*r**2-a**2*d**2+(2*e*a*b-2*a*c)*d-c**2+2*e*b*c-e**2*b**2)))/(b**2+a**2)-(a*b*d)/(b**2+a**2)-(b*c)/(b**2+a**2)-(e*a**2)/(b**2+a**2)
        '''
        return (x1, y1), (x2, y2)

def get(img, x):
    mask = cv2.inRange(img, x, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if len(cnt) > 20 and radius > 5:
            results.append((int(x), int(y), int(radius)))
    results.sort(key=lambda c: c[2], reverse=True)
    return results #print(1101fbdbsbjksdjnnjdbnjbjfdbjbdfbj)

def create(clr, maxx, nam, drw=True, sho=False):
    clr = np.array(clr, dtype=np.uint8)
    msk = cv2.inRange(img, clr, clr)
    cnt, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lst = []
    for cnt in cnt:
        if cv2.contourArea(cnt) > 0:
            (x, y), rad = cv2.minEnclosingCircle(cnt)
            lst.append((int(x), int(y), int(rad), cv2.contourArea(cnt)))
    lst.sort(key=lambda it: it[3], reverse=True)
    lst = lst[:maxx]
    shp = [(x, y, r) for (x, y, r, a) in lst]
    if sho:
        col = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
        for (x, y, r) in shp:
            cv2.circle(col, (x, y), r, clr.tolist()[::-1], -1)
            cv2.circle(col, (x, y), 3, (64, 98, 255), -1)
        cv2.imshow(f"{nam.capitalize()} prox", col)
    if drw:
        for (x, y, r) in shp:
            cv2.circle(img, (x, y), 3, (64, 98, 255), -1)
    return shp


def f(a, b):
    while abs(a - b) > 180:
        if a < b:
            a += 360
        else:
            b += 360
    return b - a

def wait(x):
    motors(0, 0, 0)
    motors(0, 0, 0)
    t0 = time.time()
    while t0 + x > time.time():
        photo()
        cv2.imshow("img", img)
        cv2.waitKey(1)

import numpy as np

import numpy as np


def proc7(img):
    col = [e[::-1] for e in [[138,33,80],[28,73,146],[30,114,99], [167,85,3],[151,151,151],[0,0,0], [77,74,70], [119,118,111]]]
    lb = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb[:,:,1:3] -= 128
    arr = np.array(col, np.uint8)
    lbc = cv2.cvtColor(arr.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)
    lbc[:,1:3] -= 128
    h, w, _ = lb.shape
    d = lb.reshape(h, w, 1, 3) - lbc.reshape(1, 1, len(col), 3)
    d = np.sum(d*d, axis=3)
    k = np.argmin(d, axis=2)
    out = np.empty((h, w, 3), np.uint8)
    for i in range(len(col)):
        out[k==i] = col[i]
    return out, [e for e in col]


def analyze():
    global img, phase, data, M
    img, colors = proc7(img)
    pink_shapes = create(colors[0], 2, "pink")
    blue_shapes = create(colors[1], 1, "blue")
    green_shapes = create(colors[2], 2, "green")
    orange_shapes = create(colors[3], 1, "orange")
    if len(pink_shapes) != 2 or len(blue_shapes) != 1 or len(orange_shapes) != 1:
        motors(0, 0, 0)
        return
    O = (blue_shapes[0][0], blue_shapes[0][1])
    A = ((pink_shapes[0][0] + pink_shapes[1][0]) // 2, (pink_shapes[0][1] + pink_shapes[1][1]) // 2)
    #B = ((green_shapes[0][0] + green_shapes[1][0]) // 2, (green_shapes[0][1] + green_shapes[1][1]) // 2)
    C = (orange_shapes[0][0], orange_shapes[0][1])
    D = (2, int(h // 2))
    E = (2, int(h // 3))
    F = (2, int(h - h // 3))
    v = Vector(C[0] - D[0], C[1] - D[1])
    if M == (-1, -1): M = (D[0] + int((C[0] - D[0]) * 100 / v.len()), D[1] + int((C[1] - D[1]) * 100 // v.len()))
    print(M)
    R = 150
    cv2.line(img, O, A, (3, 250, 155), 2)
    #cv2.line(img, O, B, (251, 96, 3), 2)
    l = Line(D, C)
    G, H = l.intersect(C[0], C[1], R)
    cv2.line(img, O, C, (232, 127, 179), 2)
    cv2.line(img, E, F, (230, 230, 101), 2)
    cv2.line(img, C, D, (118, 232, 186), 2)
    cv2.circle(img, C, R, (220, 220, 220), 2)
    cv2.circle(img, M, 3, (64, 98, 255), -1)
    G, H = (int(G[0]), int(G[1])), (int(H[0]), int(H[1]))
    if Vector(G[0] - D[0], G[1] - D[1]).len() < Vector(H[0] - D[0], H[1] - D[1]).len():
        G, H = H, G
    x, y = C[0], C[1]
    c, d = G[0] -x, G[1] - y
    points = [G, (x - d, y + c), (x + d, y - c)]
    for i in points:
        cv2.circle(img, i, 3, (64, 98, 255), -1)
    
    if phase // 2 == 0:
        if O[0] > C[0]:
            phase = 6
        elif Vector(points[1][0] - O[0], points[1][1] - O[1]).len() < Vector(points[2][0] - O[0], points[2][1] - O[1]).len():
            phase = 2
        else:
            phase = 4
        return
    elif phase // 2 == 1:
        target = points[1]
    elif phase // 2 == 2:
        target = points[2]
    elif phase // 2 == 3:
        target = points[0]
    elif phase // 2 == 4:
        target = M
    else:
        motors(0, 0, 0)
        print("Stop")
        return
    
    cv2.line(img, O, target, (20, 20, 255), 2)
    v1 = Vector(A[0] - O[0], A[1] - O[1])
    v2 = Vector(target[0] - O[0], target[1] - O[1])
    angle = abs(min((v1.angle() - v2.angle()) % 360, 360 - (v1.angle() - v2.angle()) % 360))
    bangle = angle < 10
    dist = int(v2.len())
    bdist = dist < 25
    print("dist =", dist)
    cv2.putText(img, f"{int(v1.angle())} {int(v2.angle())} {int(angle)} {int(v2.len())}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (228, 50, 228), 1, 1)
    print("phase =", phase, end=" ")
    if phase > 0 and phase % 2 == 0:
        if bangle:
            motors(0, 0, 0)
            wait(1)
            phase += 1
            return
        snow = bpovmin + (bpovmax - bpovmin) * (angle / 180)
        print(int(snow), int(angle), int(v1.angle()), int(v2.angle()), int(v1.angle() - v2.angle()), int((v1.angle() - v2.angle()) % 360))
        if (v1.angle() - v2.angle()) % 360 < 180:
            print("R")
            motors(-snow, -snow, -snow)
        else:
            print("L")
            motors(snow, snow, snow)
    elif phase > 0 and phase % 2 == 1:
        #motors(0, 0, 0)
        print("F")
        if bdist:
            motors(0, 0, 0)
            wait(1)
            if phase == 3:
                phase += 2
            phase += 1
            return
        delt = (bmovemax - bmovemin) / 2 * ((f(v1.angle(), v2.angle())) / 20)
        snow = bmovemin + (bmovemax - bmovemin) / 2 * (dist) / max(dist, 100)
        print(snow, int(delt), f(v1.angle(), v2.angle()), v1.angle(), v2.angle())
        motors(snow + delt, 0, -(snow - delt))
    else:
        print("Stop")
        motors(0, 0, 0)

def main():
    motors(0, 0, 0)
    time.sleep(5)
    photo()
    cv2.imwrite('output.png', img)
    analyze()
    data = []
    t0 = time.time()

    while True:
        photo()
        analyze()
        data.append(time.time())
        while len(data) > 1 and data[-1] - data[0] > 10:
            del data[0]
        print(len(data) / 10)
        cv2.imshow("img", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        motors(0, 0, 0)
        exit(0)