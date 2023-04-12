
import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')
from matplotlib import pyplot as plt
import math


frame_in_w = 1366
frame_in_h = 768

videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);


while True:

    ret ,frame_vga =videoIn.read()
    cv2.imshow('frame' ,frame_vga)

    cols ,rows = frame_vga.shape[1] ,frame_vga.shape[0]

    cv2.line(frame_vga ,(cols//2 ,0) ,(cols//2 ,rows) ,(0 ,255 ,0) ,1)
    cv2.line(frame_vga ,(0 ,rows//2) ,(cols ,rows//2) ,(0 ,255 ,0) ,1)
    cv2.putText(frame_vga, " (0,0) ", (cols//2 ,rows// 2 +20) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(0 ,0 ,0), 2, cv2.LINE_4)

    hsv = cv2.cvtColor(frame_vga, cv2.COLOR_BGR2HSV)  # hsv: hue saturation value

    lower_yellow =np.array([22, 50, 50])  # [15,90,100]       #[15,100,80]                    #ffef2a    15,83,100
    upper_yellow = np.array([90, 255, 255])  # [100, 255, 255]   #[100,255,220]

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    result = cv2.bitwise_and(frame_vga, frame_vga, mask=mask)

    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    ret_t_value, thresh_convert = cv2.threshold(result_gray, 80, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh_convert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:

        area = cv2.contourArea(c)

        if (area < 800):
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)

        print("x = ", x, "y = ", y)
        center = (int(x), int(y))
        radius = int(radius)

        print("Radius = ", radius)

        center1 = [0, 0]

        if center[0] > cols // 2 and center[1] < rows // 2:  # 1st quadrant
            center1[0] = center[0] - cols // 2
            center1[1] = rows // 2 - center[1]
        elif center[0] < cols // 2 and center[1] < rows // 2:  # 2nd quadrant
            center1[0] = center[0] - cols // 2
            center1[1] = rows // 2 - center[1]

        elif center[0] < cols // 2 and center[1] > rows // 2:  # 3rd quadrant
            center1[0] = center[0] - cols // 2
            center1[1] = rows // 2 - center[1]


        elif center[0] > cols // 2 and center[1] > rows // 2:  # 4th quadrant
            center1[0] = center[0] - cols // 2
            center1[1] = rows // 2 - center[1]

        print("Center = ", center)
        print("Center1 = ", center1)

        center = (center[0], center[1])
        cv2.circle(frame_vga, center, radius, (0, 255, 0), 2)
        cv2.putText(frame_vga, str(center1[0]) + "," + str(center1[1]), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2, cv2.LINE_4)
        cv2.line(frame_vga, (cols // 2, rows // 2), (center[0], center[1]), (0, 0, 255), 2)

        dy, dx = rows // 2 - center[1], center[0] - cols // 2
        if dx == 0 and dy != 0:
            angle_deg = 90
        elif dx == 0 and dy == 0:
            angle_deg = 0
        else:
            angle = math.atan(dy / dx)

            angle_deg = round((180 / np.pi) * angle, 2)

        dist = round(math.sqrt(((center1[0]) ** 2) + ((center1[1]) ** 2)), 2)

        # cv2.putText(frame_vga,"Distance: "+ str(dist),(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,200),2,cv2.LINE_4)
        # cv2.putText(frame_vga,"Angle: "+ str(angle_deg) + " degrees",(20,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,200),2,cv2.LINE_4)
        print("angle = ", angle_deg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






