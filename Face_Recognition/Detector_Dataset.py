import cv2
import time
from datetime import datetime
import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="face_recognition"
)
tgl = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
recog = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create(); 
recognizer.read("recognizer/traindata.yml")
ID=0
mysql = db.cursor()
font = cv2.FONT_HERSHEY_SIMPLEX
while 1:
    frame = camera.read()[1]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = recog.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        ID,conf=recognizer.predict(gray[y:y+h,x:x+w])
        mysql.execute("select * from face_dataset where ID="+str(ID))
        cv2.putText(frame,mysql.fetchall()[0][1]+"  "+str(conf)[:-10],(x,y-10),font,0.5,(255,0,0),2)
        if conf <= 60:
            time.sleep(0.5)
            cv2.imwrite("datacapture/ok"+str(conf)+".jpg",frame[y:y+h,x:x+w])
    cv2.imshow("",frame)
    if cv2.waitKey(1) == 27:
        break;
camera.release()
cv2.destroyAllWindows()
        
