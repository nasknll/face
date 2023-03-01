import cv2
import numpy as np

def face_detection(filename):
    facer = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    mouth = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_mouth.xml')
    nose = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_nose.xml')
    
    #第二步，导入人脸识别的图片并将其灰度化
    #img = cv2.imread('./Fjum2UHWIAYQxop.jpg')
    img = cv2.imread(filename)
    
    #第三步，进行人脸识别
    #[[x,y,w,h]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #检测出的人脸上再检测眼睛
    faces = facer.detectMultiScale(gray, 1.1, 3)
    i = 0
    j = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_img = img[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(roi_img, 1.1, 3)
        for (x,y,w,h) in eyes:
            cv2.rectangle(roi_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_eye=roi_img[y:y+h, x:x+w]
            eyename = 'eye' + str(j)
            j = j+1 
            cv2.imshow(eyename, roi_eye)
    
        i = i+1
        winname = 'face' + str(i)
        cv2.imshow(winname, roi_img)
    
    
    # mouths = mouth.detectMultiScale(gray, 1.1, 3)
    # for (x,y,w,h) in mouths:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # noses = nose.detectMultiScale(gray, 1.1, 3)
    # for (x,y,w,h) in noses:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
    
    cv2.imshow('img', img)
    
    cv2.waitKey()

#face_detection("./p2684182_e_h9_ab.jpg")

print ("hello world")

cap = cv2.VideoCapture(0)

# cv2.capture

while(True):
    ret, frame = cap.read()
    
    cv2.imwrite("output.jpg", frame)
    face_detection("./output.jpg")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()



























