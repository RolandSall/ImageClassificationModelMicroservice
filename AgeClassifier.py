import json
import os
from flask import Flask, request, redirect, url_for, flash, jsonify
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
import os, sys
import sys
import cv2
import dlib
import imutils
import pickle
import json
import numpy as np
import pandas as pd
import sklearn
import skopt
import matplotlib.pyplot as plt
from imutils import face_utils
from math import atan2, degrees

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "C:\\Users\\user\\PycharmProjects\\ImageClassificationMicroservice\\shape_predictor_68_face_landmarks.dat")


def Binarypattern(im):
    img = np.zeros_like(im)
    n = 3  # taking kernel of size 3*3
    for i in range(0, im.shape[0] - n):  # for image height
        for j in range(0, im.shape[1] - n):  # for image width
            x = im[i:i + n, j:j + n]  # reading the entire image in 3*3 format
            center = x[1, 1]  # taking the center value for 3*3 kernel
            img1 = (
                           x >= center) * 1.0  # checking if neighbouring values of center value is greater or less than center value
            img1_vector = img1.T.flatten()  # getting the image pixel values
            img1_vector = np.delete(img1_vector, 4)
            digit = np.where(img1_vector)[0]
            if len(digit) >= 1:  # converting the neighbouring pixels according to center pixel value
                num = np.sum(2 ** digit)  # if n> center assign 1 and if n<center assign 0
            else:  # if 1 then multiply by 2^digit and if 0 then making value 0 and aggregating all the values of kernel to get new center value
                num = 0
            img[i + 1, j + 1] = num
    return img


def rotateFace(img):
    points = []

    imgH, imgW, imgC = img.shape

    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:

        jaw = []
        right_eyebrow = []
        left_eyebrow = []

        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            landmarks = predictor(image=gray, box=face)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == "jaw" or name == "right_eyebrow" or name == "left_eyebrow":
                for n in range(i, j):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    locals()[name].append([x, y])

        xL, yL, wL, hL = cv2.boundingRect(np.array(left_eyebrow))
        xR, yR, wR, hR = cv2.boundingRect(np.array(right_eyebrow))

        eye_left_center = [(2 * xL + wL) / 2, (2 * yL + hL) / 2]
        eye_right_center = [(2 * xR + wR) / 2, (2 * yR + hR) / 2]

        if eye_left_center[0] >= eye_right_center[0]:
            xDiff = eye_left_center[0] - eye_right_center[0]
            yDiff = eye_left_center[1] - eye_right_center[1]
            angle = degrees(atan2(yDiff, xDiff))
        else:
            xDiff = eye_right_center[0] - eye_left_center[0]
            yDiff = eye_right_center[1] - eye_left_center[1]
            angle = degrees(atan2(yDiff, xDiff))

        # (x, y, w, h) = cv2.boundingRect(np.array([[x1,y1],[x2,y2]]))

        # roi = img[y: y + h , x : x + w ]
        rotated = imutils.rotate(img, angle)
        return rotated
    return []


def mid_face_detection_face_detector(img):
    points = []
    right_eye = []
    left_eye = []
    imgH, imgW, imgC = img.shape

    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    roi = []
    faces = detector(gray)
    if (faces):
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            landmarks = predictor(image=gray, box=face)

        imgH, imgW, imgC = img.shape
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if (name == "right_eye" or name == "left_eye"):
                for n in range(i, j):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    points.append([x, y])
                    locals()[name].append([x, y])

        (x, y, w, h) = cv2.boundingRect(np.array([points]))

        addTop = int(imgH * 0.17)
        addBottom = int(imgH * 0.08)
        addLeft = int(imgW * 0.06)
        addRight = int(imgW * 0.06)

        if x < addLeft:
            addLeft = x
        if (imgW - x) < addRight:
            addRight = imgW - x + (addRight - x)
        if y < addTop:
            addTop = y
        roi = img[y - addTop:y + h + addBottom, x - addLeft: x + w + addRight]

        return roi
    return []


model_path = 'models/svm-age-3categories-Faces.sav'
scaler_path = 'scalars/scalerFacesMidFace.pkl'
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/predict', methods=['POST'])
def predict():
    img_path = request.json
    print(img_path)
    dummyPath = "C:\\Users\\user\\IdeaProjects\\imageclassificationbackend\\testing.jpg"
    img = cv2.imread(img_path)
    rotated = rotateFace(img)
    rotated = cv2.resize(rotated, (460, 460), interpolation=cv2.INTER_AREA)
    rotated = cv2.GaussianBlur(rotated, (3, 3), 0)
    roi = mid_face_detection_face_detector(rotated)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    imgLBP = Binarypattern(eq)
    cv2.imshow("asd", imgLBP)
    cv2.waitKey(0)
    vectorLBP = imgLBP.flatten()
    freq, lbph, _ = plt.hist(vectorLBP, bins=2 ** 8)
    freq = scaler.fit_transform([freq])
    pred = model.predict(freq)
    if pred == 0:
        data_message = {"output": "Middle-Age"}
        print(data_message)
        return json.dumps(data_message)
    if pred == 2:
        data_message = {"output": "Young"}
        print(data_message)
        return json.dumps(data_message)
    if pred == 1:
        data_message = {"output": "Old"}
        print(data_message)
        return json.dumps(data_message)


if __name__ == "__main__":
    app.run()
