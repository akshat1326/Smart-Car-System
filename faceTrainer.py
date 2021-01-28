from PIL import Image
import os
import cv2 as cv
import numpy as np
import time
import json


detector = cv.CascadeClassifier("C:\\Users\\jtuli\\AppData\\Roaming\\Python\\Python37\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")
recognizer = cv.face.LBPHFaceRecognizer_create()
path = ("C:/Users/jtuli/Desktop/face detection/drivers/") #Images of all the drivers will be stored in the 'drivers' folder
#num_users = input("Enter the number of drivers to be registered: ")
usernames_list = [] # Is a list of driver ID's ( "1", "2", .... )
driver_dict = {}

i = 0
while 1:
    key = input("Register? type Yes/No: ")
    if(key.upper() == "NO" ):
        break
    print("Registration "+str(i+1))
    usernames_list.append(str(i+1))
    driver_name = input("Enter name: ")
    driver_dict[str(i+1)] = driver_name
    i+=1
    

faceSamples = [] #sample of faces as a list of numpy arrays
Ids = [] #ID corresponding to each face sample
print(usernames_list)
print(driver_dict)

for i in usernames_list: # retrieving face samples of the ith person
    this_path = path + i
    ID = int(i)

    for pic_file in os.listdir(this_path):
        #imagePaths.append(tmp + "/" + pic_file) 
        grayscale_image = Image.open(this_path + "/" + pic_file).convert('L') #We convert the picture into grayscale
        image_as_numpyarray = np.array(grayscale_image, 'uint8')
        # extract the face from the training image sample
        faces_list = detector.detectMultiScale(image_as_numpyarray) #faces_list is a list of faces detected in this pic
        for (x, y, w, h) in faces_list:
            face_as_numpyarray = image_as_numpyarray[y:y + h, x:x + w] #Numpy array slicing that extracts the face portion of the array
            faceSamples.append(face_as_numpyarray)  
            Ids.append(ID)


json_string = json.dumps(driver_dict) #storing the names of drivers
f = open("driver_names.json", "w")
f.write(json_string)
f.close()

print(Ids)
print(len(faceSamples))
recognizer.train(faceSamples, np.array(Ids))
recognizer.save("C:\\Users\\jtuli\\Desktop\\face detection runs\\trained21.yml")
