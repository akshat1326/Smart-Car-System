import cv2 as cv
import numpy as np
import json
from playsound import playsound
import time
import winsound

detector = cv.CascadeClassifier("C:\\Users\\jtuli\\AppData\\Roaming\\Python\\Python37\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")
recognizer = cv.face.LBPHFaceRecognizer_create()

f = open("driver_names.json", "r")
driver_dict = json.load(f)
f.close()
#driver_dict = {"1": "tulika", "2":"omkar", "3":"akshat", "4": "avinash"}


print(driver_dict)

cam_in = cv.VideoCapture(0)

recognizer.read("C:\\Users\\jtuli\\Desktop\\face detection runs\\trained21.yml")
path = ""

while 1:
    ret, img = cam_in.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h),(255,255,0),2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if str(id) in driver_dict:
            name = driver_dict[str(id)]
            path += "C:\\Users\\jtuli\\Downloads\\Journey - Don't Stop Believin' (Audio).mp3"
        else:
            name = "Access Denied"
            path += "C:\\Users\\jtuli\\Downloads\\Taylor Swift - Delicate (Lyrics).mp3"
        cv.putText(img, name, (x, y+h), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    #winsound.PlaySound(path, winsound.SND_ASYNC)
    #playsound(path)
    cv.imshow("Face",img)
    #time.sleep(15)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cam_in.release()
cv.destroyAllWindows()
