import cv2
from datetime import datetime
import time
import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="face_recognition"
)
value = 0
tgl = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
recog = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
Nama = str(input("Input Name : "))
ID = int(input("Input ID : "))
mysql = db.cursor()
mysql.execute("select * from face_dataset where ID="+str(ID))
try:
    if ID in mysql.fetchall()[0]:
        pass
    else:
        mysql.execute("INSERT INTO face_dataset (ID,Nama,Capture) Values ("+str(ID)+",'"+Nama+"','"+str(tgl)+"')")
except:
    mysql.execute("INSERT INTO face_dataset (ID,Nama,Capture) Values ("+str(ID)+",'"+Nama+"','"+str(tgl)+"')")
db.commit()
while 1:
    frame = camera.read()[1]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = recog.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        value += 1
        cv2.imwrite("dataset/"+str(tgl).replace(":","-")+"-"+Nama+"-"+str(ID)+"-"+str(value)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),5)
    cv2.imshow("",frame)
    cv2.waitKey(1)
    if value >= 20:
        break
cv2.destroyAllWindows()
        
