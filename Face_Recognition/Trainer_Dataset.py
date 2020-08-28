import os
import cv2
import numpy as np

recog = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

def getImage(path):
    imagepath=[os.path.join(path,i) for i in os.listdir(path)]
    face=[]
    ID=[]
    for i in imagepath:
        faceim = cv2.imread(i,0)
        face_numpy = np.array(faceim,"uint8")
        ids=int(os.path.split(i)[-1].split('-')[7])
        face.append(face_numpy)
        ID.append(ids)
        cv2.waitKey(10)
    return ID,face

Ids,faces=getImage(path)
recog.train(faces,np.array(Ids))
recog.save("recognizer/traindata.yml")
cv2.destroyAllWindows()
