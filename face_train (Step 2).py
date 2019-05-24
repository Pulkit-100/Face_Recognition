import os
import numpy as np
from PIL import Image
import cv2

recog=cv2.face.LBPHFaceRecognizer_create()
path="DATASET"

def getImageswithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    try:        
        for imgpath in imagePaths:
            faceImg= Image.open(imgpath).convert("L")
            faceNp=np.array(faceImg,'uint8')
            ID=os.path.split(imgpath)[-1].split()[0]
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow("training",faceNp)
            cv2.waitKey(10)
    except Exception as e:
        print(e)
    d={}
    c=1
    for pk in IDs:
        if pk not in d:
            d[pk]=c
            c+=1
    with open("Names.txt","w") as fp:
        for i in d:
            fp.write(str(str(d[i])+"-"+i+"\n"))

    for pk in range(len(IDs)):
        IDs[pk]=d[IDs[pk]]
    return IDs, faces

IDs,faces = getImageswithID(path)
recog.train(faces,np.array(IDs))
recog.write("trainingData.yml")
cv2.destroyAllWindows()
