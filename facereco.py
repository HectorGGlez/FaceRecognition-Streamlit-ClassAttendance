import streamlit as st
import pandas as pd
import numpy as np
import cv2
import face_recognition
import os
import sys
from pathlib import Path
from datetime import datetime

st.title('Face RECOGNITION')

index = st.sidebar.selectbox(
    'Toma lista',
    (0, 1, 2)
)
lista = ["/Users/hectorgonzalez/Documents/CLOUD/streamlit/Video/Josue.mp4",
         "/Users/hectorgonzalez/Documents/CLOUD/streamlit/Video/rudy.mp4", "/Users/hectorgonzalez/Documents/CLOUD/streamlit/Video/video.mp4"]


st.write(f'You selected: {lista[index]}')

path = "ImagesAttendance"
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


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}, {now}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Videos sections
# Rudys one /Users/hectorgonzalez/Documents/CLOUD/streamlit/Video/vid.mp4

videoLoaded = (
    lista[index])

video_file = open(
    videoLoaded, 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

cap = cv2.VideoCapture(videoLoaded)

while True:
    success, img = cap.read()
    if success == False:
        print("No image")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    #imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(
            encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(
            encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        markAttendance(name)
        print(name)
        st.error(f"Lista de alumnos {classNames}", icon="🚨")
        st.success(name, icon="✅")

    cv2.imshow('Webcam', img)
    cv2.waitKey()
