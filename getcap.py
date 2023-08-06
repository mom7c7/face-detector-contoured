import cv2
import numpy as np
from datetime import *
from time import sleep

face_cascade = cv2.CascadeClassifier('face_detector.xml')
counter = 0
cap = cv2.VideoCapture(0)
while True:
    timee = datetime.now()
    sttime = str(timee) 
    ret , img = cap.read()
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(gray_img,kernel,iterations = 1)
    _, thresh = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)
    for i in img_contours:
        if cv2.contourArea(i) > 150000:
            break
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [i],-1, 255, -1)
    new_img = cv2.bitwise_and(img, img, mask=mask)
    faces = face_cascade.detectMultiScale(new_img,1.1,4)
    for (x, y, w, h) in faces: 
        if True:
            cv2.rectangle(new_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            counter +=1
        if x > (640/4) and x < ((640/4)*3) and y > (480/4) and y < ((480/4)*3) and len(faces) != None and counter > 0:
            jtimee = sttime.split(' ')
            jtime = jtimee[1]
            sbts = jtime.split(':')
            sbts2 = sbts[2]
            sbts3 = sbts2.split('.')
            allsbts = sbts3[0]+sbts3[1]
            oneandtwo = sbts[0] + sbts[1] + allsbts
            name = oneandtwo+'.png'
            cv2.imwrite(name,new_img)
            sleep(1)
        cv2.imshow("Cam",new_img)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()    