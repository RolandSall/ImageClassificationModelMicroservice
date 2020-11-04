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

### Directory to print the images that will be sent to the front-end
DIRC = "C:\\Users\\user\\WebstormProjects\\image_classification_front_end\\src\\Commun\\resources\\imagesLBP\\"

### Detector and predictor for facial detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "C:\\Users\\user\\PycharmProjects\\ImageClassificationMicroservice\\shape_predictor_68_face_landmarks.dat")


### To extract the model and scaler in case they are saved in the model
class full_pipelined_model(object):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler


### LBP method
def Binarypattern(im):
    img = np.zeros_like(im)
    n = 3  # taking kernel of size 3*3
    for i in range(0, im.shape[0] - n):  # for image height
        for j in range(0, im.shape[1] - n):  # for image width
            x = im[i:i + n, j:j + n]  # reading the entire image in 3*3 format
            center = x[1, 1]  # taking the center value for 3*3 kernel
            img1 = (
                           x >= center) * 1.0  # Checking if neighbouring values of center value is greater or less than center value
            img1_vector = img1.T.flatten()  # Getting the image pixel values
            img1_vector = np.delete(img1_vector, 4)
            digit = np.where(img1_vector)[0]
            if len(digit) >= 1:  # Converting the neighbouring pixels according to center pixel value
                num = np.sum(2 ** digit)  # if n> center assign 1 and if n<center assign 0
            else:  # if 1 then multiply by 2^digit and if 0 then making value 0 and aggregating all the values of kernel to get new center value
                num = 0
            img[i + 1, j + 1] = num
    return img


### Method to center the image in the face is not centered at the middle
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


### Method to rotate the face for better detection
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


### Method to detect the full face
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


def mid_face_detection_face_detector(img):
    points = []
    right_eye = []
    left_eye = []
    ImgH, ImgW, imgC = img.shape
    kernel = np.ones((3, 3), np.float32) / 7
    gf = cv2.filter2D(img, -1, kernel)
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    roi = []
    faces = detector(gray)
    if faces:
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            landmarks = predictor(image=gray, box=face)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            for n in range(i, j):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                points.append([x, y])

        (x, y, w, h) = cv2.boundingRect(np.array([points]))

        addTop = int(h * 0.1)
        addBottom = int(h * 0.1)
        addLeft = int(w * 0.1)
        addRight = int(w * 0.1)

        if addTop > y:
            addTop = 0
        if addLeft > x:
            addLeft = 0
        if addRight > ImgW - x - w:
            addRight = ImgW - x - w
        if addBottom > ImgH - y - h:
            addBottom = ImgH - y - h

        roi = gf[y - addTop: y + h + addBottom, x - addLeft: x + w + addRight]

        roi = cv2.resize(roi, (460, 460), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(src=roi, code=cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        points = []
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            landmarks = predictor(image=gray, box=face)

        ld = ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow"]
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name in ld:
                for n in range(i, j):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    points.append([x, y])

        (x, y, w, h) = cv2.boundingRect(np.array([points]))
        addBottom = int(h * 0.9)

        roi = roi[y:y + h + addBottom, x: x + w]
        roi = cv2.resize(roi, (200, 100), interpolation=cv2.INTER_AREA)

        return roi

    return []


def mouth_detection_face_detector(img):
    points = []
    right_eye = []
    left_eye = []
    ImgH, ImgW, ImgC = img.shape
    kernel = np.ones((3, 3), np.float32) / 7
    gf = cv2.filter2D(img, -1, kernel)
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    roi = []
    faces = detector(gray)
    if faces:
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            landmarks = predictor(image=gray, box=face)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            for n in range(i, j):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                points.append([x, y])

        (x, y, w, h) = cv2.boundingRect(np.array([points]))

        addTop = int(h * 0.1)
        addBottom = int(h * 0.1)
        addLeft = int(w * 0.1)
        addRight = int(w * 0.1)

        if addTop > y:
            addTop = 0
        if addLeft > x:
            addLeft = 0
        if addRight > ImgW - x - w:
            addRight = ImgW - x - w
        if addBottom > ImgH - y - h:
            addBottom = ImgH - y - h

        roi = gf[y - addTop: y + h + addBottom, x - addLeft: x + w + addRight]

        roi = cv2.resize(roi, (460, 460), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(src=roi, code=cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        points = []

        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            landmarks = predictor(image=gray, box=face)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name in "mouth":
                for n in range(i, j):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    points.append([x, y])

        (x, y, w, h) = cv2.boundingRect(np.array([points]))
        addBottom = int(h * 1)
        addTop = int(h * 0.7)
        addRight = int(w * 0.4)
        addLeft = int(w * 0.4)

        roi = roi[y - addTop:y + h + addBottom, x - addLeft: x + w + addRight]
        roi = cv2.resize(roi, (200, 100), interpolation=cv2.INTER_AREA)

        return roi

    return []


def lbp_freq(roi):
    gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray_img)
    eq = cv2.equalizeHist(eq)
    eq = cv2.equalizeHist(eq)
    imgLBP = Binarypattern(eq)  # calling the LBP function using gray image
    vectorLBP = imgLBP.flatten()  # for histogram using the vector form of image pixels

    freq, lbp, _ = plt.hist(vectorLBP, bins=2 ** 8)
    return gray_img, eq, imgLBP, freq


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
with open('full_face_knn.pkl', 'rb') as input:
    ffMS = pickle.load(input)

with open('mid_face_knn.pkl', 'rb') as input:
    mfMS = pickle.load(input)

with open('mouth_knn.pkl', 'rb') as input:
    mMS = pickle.load(input)


### Flask API to call the predictor
@app.route('/predict', methods=['POST'])
def predict():
    img_path = request.json
    print(img_path)
    img = cv2.imread(img_path)
    img = centerImage(img)
    imgR = cv2.resize(img, (460, 460), interpolation=cv2.INTER_AREA)
    cv2.imwrite(  # Send the resized imaged to Front-End
        "%sresize.jpg" % DIRC, imgR)
    rotated = rotateFace(imgR)
    if len(rotated) == 0:
        rotated = rotateFace(img)

    cv2.imwrite(
        "%srotate.jpg" % DIRC, rotated)
    ff = full_face_detection_face_detector(rotated)
    mf = mid_face_detection_face_detector(rotated)
    m = mouth_detection_face_detector(rotated)

    gray_imgFF, eqFF, imgLBPFF, lbpff = lbp_freq(ff)
    cv2.imwrite(
        "%sgrayFF.jpg" % DIRC, gray_imgFF)
    cv2.imwrite(
        "%seqFF.jpg" % DIRC, eqFF)
    cv2.imwrite(
        "%sLBPFF.jpg" % DIRC, imgLBPFF)
    gray_imgMF, eqMF, imgLBPMF, lbpmf = lbp_freq(mf)
    cv2.imwrite(
        "%sgrayMF.jpg" % DIRC, gray_imgMF)
    cv2.imwrite(
        "%seqMF.jpg" % DIRC, eqMF)
    cv2.imwrite(
        "%sLBPMF.jpg" % DIRC, imgLBPMF)
    gray_imgM, eqM, imgLBPM, lbpm = lbp_freq(m)
    cv2.imwrite(
        "%sgrayM.jpg" % DIRC, gray_imgM)
    cv2.imwrite(
        "%seqM.jpg" % DIRC, eqM)
    cv2.imwrite(
        "%sLBPM.jpg" % DIRC, imgLBPM)

    Xff = ffMS.scaler.transform([lbpff])
    Xmf = mfMS.scaler.transform([lbpmf])
    Xm = mMS.scaler.transform([lbpm])

    result = ffMS.model.predict_proba(Xff) * 0.266666 + mfMS.model.predict_proba(
        Xmf) * 0.0666666 + mMS.model.predict_proba(Xm) * 0.66666666

    print(result)
    resultJson = {
        "young": "{}".format((result[0][0])),
        "middleAge": "{}".format(result[0][1]),
        "old": "{}".format(result[0][2]),
        "error": ""
    }

    return json.dumps(resultJson)  # Send back the Result to the Java Service


if __name__ == "__main__":  # Main app to run flask on localhost:5000
    app.run()
