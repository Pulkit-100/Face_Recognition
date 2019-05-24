import cv2
import numpy as np

facedetect= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("trainingData.yml")

Id=0
fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale =1
fontColor=(255,0,255)

with open("Names.txt","r") as fp:
    NAMES=fp.readlines()
Names={}
for i in NAMES:
    Names[int(i[0])]=i[2:]
    
#print(NAMES)
cam=cv2.VideoCapture(0)

while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        Id,conf=rec.predict(gray[y:y+h,x:x+w])
        if conf<30:
            cv2.putText(img,"Not Found !! ",(x,y+h),fontFace,fontScale,fontColor)
        else:
            cv2.putText(img,str(Names[Id])+' '+str(conf),(x,y+h),fontFace,fontScale,fontColor)
    cv2.imshow("face",img)
    if (cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
