import json
import os
from math import atan2, degrees
import cv2
import dlib
import pickle
import matplotlib.pyplot as plt
import imutils
import numpy as np
from flask import Flask, request
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "C:\\Users\\user\\PycharmProjects\\ImageClassificationMicroservice\\shape_predictor_68_face_landmarks.dat")


class full_pipelined_model(object):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler


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


def full_face_detection_face_detector(img):
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    jaw = []
    right_eyebrow = []
    left_eyebrow = []
    if len(faces) > 0:
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
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    locals()[name].append([x, y])

        kernel = np.ones((3, 3), np.float32) / 18
        gf = cv2.filter2D(img, -1, kernel)

        pts = []

        del right_eyebrow[0]
        right_eyebrow = right_eyebrow[::-1]

        del left_eyebrow[0]
        left_eyebrow = left_eyebrow[::-1]

        pts = jaw + left_eyebrow + right_eyebrow
        pts = np.array(pts)

        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        cropped = gf[y:y + h, x:x + w].copy()
        pts = pts - pts.min(axis=0)

        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        bg = np.ones_like(cropped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst2 = bg + dst

        return dst2
    return []


def centerImage(img):
    points = []
    right_eye = []
    left_eye = []
    imgH, imgW, imgC = img.shape

    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    roi = []
    faces = detector(gray)
    try:
        faces[0]
    except:
        return {"error": "could not detect age"}

    if faces:
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            landmarks = predictor(image=gray, box=face)

        imgH, imgW, imgC = img.shape
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == "right_eye" or name == "left_eye":
                for n in range(i, j):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    points.append([x, y])

        (x, y, w, h) = cv2.boundingRect(np.array([points]))
        width = imgH / 1.25
        add = int((width - w) / 2)
        left_edge = x
        right_edge = imgW - x - w
        eff = min(left_edge, right_edge)
        if add > eff:
            add = eff
        return img[0: imgH, x - add: x + w + add]


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
with open('./models/full_face_knn (2).pkl', 'rb') as input:
    pp = pickle.load(input)


@app.route('/predict', methods=['POST'])
def predict():
    img_path = request.json
    print(img_path)
    # dummyPath = "C:\\Users\\user\\IdeaProjects\\imageclassificationbackend\\testing.jpg"
    img = cv2.imread(img_path)
    try:
        img = centerImage(img)
        img = cv2.resize(img, (460, 460), interpolation=cv2.INTER_AREA)

        rotated = rotateFace(img)
        roi = full_face_detection_face_detector(rotated)
        roi = cv2.resize(roi, (460, 460), interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.equalizeHist(gray_img)
        gray_img = cv2.equalizeHist(gray_img)
        gray_img = cv2.equalizeHist(gray_img)
        imgLBP = Binarypattern(gray_img)  # calling the LBP function using gray image
        vectorLBP = imgLBP.flatten()  # for histogram using the vector form of image pixels
        # cv2.imwrite('data/dst/lena_opencv_red.jpg', vectorLBP)

    except:
        return {"error": "could not detect age"}

    # To visualize the graphs uncomment
    '''
    fig = plt.figure(figsize=(20, 8))  # sub plotting the gray, LBP and histogram
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    ax.set_title("Image")
    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(gray_img, cmap="gray")
    ax.set_title("Gray and Equalized Image")
    ax = fig.add_subplot(1, 4, 3)
    ax.imshow(imgLBP, cmap="gray")
    ax.set_title("LBP converted image")
    ax = fig.add_subplot(1, 4, 4)
    maxF = freq.max()
    ax.set_ylim(0, maxF + 1000)
    lbp = lbp[:-1]
    largeTF = freq > 5000
    # for x, fr in zip(lbp[largeTF], freq[largeTF]):
    #    ax.text(x, fr, "{:6.0f}".format(x), color="magenta")
    # ax.set_title("LBP histogram")
    # plt.show()
    '''

    freq, lbp, _ = plt.hist(vectorLBP, bins=2 ** 8)
    X = pp.scaler.transform([freq])
    result = pp.model.predict_proba(X)
    print(result)
    resultJson = {
        "young": "{}".format((result[0][0])),
        "middleAge": "{}".format(result[0][1]),
        "old": "{}".format(result[0][2]),
        "error": ""
    }

    return json.dumps(resultJson)


if __name__ == "__main__":
    app.run()
