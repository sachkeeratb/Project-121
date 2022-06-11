import cv2
from cv2 import VideoWriter_fourcc
import numpy as np
import time

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vidcap = cv2.VideoCapture('magic.api')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)

time.sleep(2)

bg = 0

frame = cv2.resize(frame, (640, 480))
image = cv2.resize(image, (640, 480))

for i in range(60):
    ret, bg = cap.read()

bg = np.flip(bg, axis = 1)

while(cap.isOpened()):
    ret, img = cap.read()
    if(not ret):
        break
    
    img = np.flip(img, axis = 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_black = np.array([104, 153, 70])
    upper_black = np.array([30, 30, 0])

    mask = cv2.inRange(hsv, lower_black, upper_black)

    lower_black = np.array([104, 153, 70])
    upper_black = np.array([30, 30, 0])


    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8()))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8()))


    res = cv2.bitwise_and(img, img, mask=mask)
    f = frame - res
    f = np.where(f==0, image, f)

    output_file.write(res)

    cv2.imshow("magic", res)
    cv2.waitKey(1)


cap.release()
output_file.release()
cv2.destroyAllWindows()