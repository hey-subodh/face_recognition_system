import cv2
import numpy as np
import face_recognition
import os

path = 'trainImages'
images = []
classNames = []
myList = os.listdir(path)

print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print(len(encodeListKnown))

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encodesCurFace = face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFace,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(235,255,255),2)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)



































# imgAmbani = face_recognition.load_image_file('trainImages/mukesh.jpg')
# imgAmbani = cv2.cvtColor(imgAmbani,cv2.COLOR_BGR2RGB)
#
#
# testimg = face_recognition.load_image_file('TestImages/gautam.jpg')
#
# testimg = cv2.cvtColor(testimg,cv2.COLOR_BGR2RGB)
#
#
# faceloc = face_recognition.face_locations(imgAmbani)[0]
# encodeAmbani = face_recognition.face_encodings(imgAmbani)[0]
# cv2.rectangle(imgAmbani,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,235),2)
#
# faceloctest = face_recognition.face_locations(testimg)[0]
# encodeTest = face_recognition.face_encodings(testimg)[0]
# cv2.rectangle(testimg,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,235),2)
#
# #At Backend we are using linear svm
#
# results = face_recognition.compare_faces([encodeAmbani],encodeTest)
# faceDis = face_recognition.face_distance([encodeAmbani],encodeTest)
# print(faceDis)  # Lower the distance, more accurate the matching
# print(results)
#
# cv2.putText(testimg,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


# cv2.imshow('Mukesh Ambani', imgAmbani)
# cv2.imshow('Mukesh ', testimg)
# cv2.waitKey(0)
