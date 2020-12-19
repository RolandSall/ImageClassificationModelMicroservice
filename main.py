import json
import pickle
from math import atan2, degrees
import cv2
import dlib
import imutils
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request
from imutils import face_utils
from tensorflow import keras
import tensorflow as tf
from skimage import transform

app = Flask(__name__)


class full_pipelined_model(object):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\User\\PycharmProjects\\ImageClassificationModelMicroservice"
                                 "\\shape_predictor_68_face_landmarks.dat")
DIRC = "C:\\Users\\User\\WebstormProjects\\ImageClassificationFrontEnd\\src\\Commun\\resources\\imagesLBP\\"


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
        right_eye = []
        left_eye = []
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            landmarks = predictor(image=gray, box=face)
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == "jaw" or name == "right_eye" or name == "left_eye":
                for n in range(i, j):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    locals()[name].append([x, y])
        xL, yL, wL, hL = cv2.boundingRect(np.array(left_eye))
        xR, yR, wR, hR = cv2.boundingRect(np.array(right_eye))
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
    ImgH, ImgW, imgC = img.shape
    kernel = np.ones((3, 3), np.float32) / 7
    gf = cv2.filter2D(img, -1, kernel)
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
        if (addTop > y):
            addTop = 0
        if (addLeft > x):
            addLeft = 0
        if (addRight > ImgW - x - w):
            addRight = ImgW - x - w
        if (addBottom > ImgH - y - h):
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
                    cv2.circle(roi, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
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


# Method to detect the full face
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
                    if (x < 0):
                        x = 0
                    if (y < 0):
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
        roi = cv2.resize(dst2, (460, 460), interpolation=cv2.INTER_AREA)
        return roi
    return []


def centerImage(img):
    points = []
    right_eye = []
    left_eye = []
    imgH, imgW, imgC = img.shape
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


def lbp_freq(roi):
    gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray_img)
    eq = cv2.equalizeHist(eq)
    eq = cv2.equalizeHist(eq)
    imgLBP = Binarypattern(eq)  # calling the LBP function using gray image
    vectorLBP = imgLBP.flatten()  # for histogram using the vector form of image pixels
    freq, lbp, _ = plt.hist(vectorLBP, bins=2 ** 8)
    return gray_img, eq, imgLBP, freq


def ANN_PP(img):
    np_image = np.array(img).astype('float32') / 255
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def processImgForANN(img_path):
    imgANN = tf.keras.preprocessing.image.load_img(
        img_path, color_mode="rgb", target_size=(150, 150), interpolation="nearest"
    )
    open_cv_image = np.array(imgANN)
    return ANN_PP(open_cv_image)


# The below part is to import our model that was trained on colab and saved as pkl file in this directory
with open('C:\\Users\\User\\PycharmProjects\\ImageClassificationModelMicroservice\\full_face_knn.pkl', 'rb') as input:
    ffMS = pickle.load(input)

with open('C:\\Users\\User\\PycharmProjects\\ImageClassificationModelMicroservice\\mid_face_knn.pkl', 'rb') as input:
    mfMS = pickle.load(input)

with open('C:\\Users\\User\\PycharmProjects\\ImageClassificationModelMicroservice\\mouth_knn.pkl', 'rb') as input:
    mMS = pickle.load(input)

mANN = keras.models.load_model(
    'C:\\Users\\User\\PycharmProjects\\ImageClassificationModelMicroservice\\ANN_INCEPTION_LBP_FULL_FACE.h5')


@app.route('/predict', methods=['POST'])
def predict():
    # Getting the PATH for the JAVA service and performing pre processing techniques such as resize, and color change
    jsonRequest = json.loads(request.json)
    ownModel = jsonRequest["ownModel"]
    img_path = jsonRequest["path"]
    print(img_path)
    img = cv2.imread(img_path)
    img = centerImage(img)
    imgR = cv2.resize(img, (460, 460), interpolation=cv2.INTER_AREA)
    cv2.imwrite("%sresize.jpg" % DIRC, imgR)
    rotated = rotateFace(imgR)
    if len(rotated) == 0:
        rotated = rotateFace(img)

    full_face_matrix_path = 'ff.png'
    # Sending a rotated picture to the Front-End resource directory
    cv2.imwrite("%srotate.jpg" % DIRC, rotated)

    # Detecting the features that we will be using
    ff = full_face_detection_face_detector(rotated)
    mf = mid_face_detection_face_detector(rotated)
    m = mouth_detection_face_detector(rotated)

    # In the below the pre process figures will be used to be sent to the interface
    # we also extract the lbp to predict
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

    cv2.imwrite(full_face_matrix_path, imgLBPFF)

    Xff = ffMS.scaler.transform([lbpff])
    Xmf = mfMS.scaler.transform([lbpmf])
    Xm = mMS.scaler.transform([lbpm])
    p = mANN.predict(processImgForANN(full_face_matrix_path))
    ANN_Result = np.array([p[0][2], p[0][0], p[0][1]])

    ## Assigning Weights based on a girdSearch to reach the most efficient feature
    result = ffMS.model.predict_proba(Xff) * 0.06667 + mfMS.model.predict_proba(Xmf) * 0.06667 + mMS.model.predict_proba(
        Xm) * 0.6667 + ANN_Result * 0.2
    print(ANN_Result[0])

    print(ownModel)
    if (ownModel=="true"):
        print(jsonRequest["clf1"],
              jsonRequest["clf2"],
              jsonRequest["clf3"])
        with open('C:\\Users\\User\\PycharmProjects\\ImageClassificationTrainerMicroService\\savedModels\\{}'.format(
                jsonRequest["clf1"]+".sav"), 'rb') as input:
            clf1 = pickle.load(input)
        with open('C:\\Users\\User\\PycharmProjects\\ImageClassificationTrainerMicroService\savedModels\\{}'.format(
                jsonRequest["clf2"]+".sav"), 'rb') as input:
            clf2 = pickle.load(input)
        with open('C:\\Users\\User\\PycharmProjects\\ImageClassificationTrainerMicroService\savedModels\\{}'.format(
                jsonRequest["clf3"]+".sav"), 'rb') as input:
            clf3 = pickle.load(input)
        ans = clf1.predict_proba([lbpm])
        print(ans[0][0])
        print(ans[0][1])
        print(ans[0][2])
        ans1 = clf2.predict_proba([lbpmf])
        print(ans1[0][0])
        print(ans1[0][1])
        print(ans1[0][2])
        ans2 = clf3.predict_proba([lbpff])
        print(ans2[0][0])
        print(ans2[0][1])
        print(ans2[0][2])

        print('----------------------------------------------')
        print(ans[0][0]*0.33 + ans1[0][0]*0.33 + ans2[0][0]*0.33)
        print(ans[0][1]*0.33 + ans1[0][1]*0.33 + ans2[0][1]*0.33)
        print(ans[0][2]*0.33 + ans1[0][2]*0.33 + ans2[0][2]*0.33)

        resultJson = {
            "youngM": "{}".format(ans[0][0]*0.33),
            "middleAgeM": "{}".format(ans[0][1]*0.33),
            "oldM": "{}".format(ans[0][2]*0.33),
            "weightM": "0.33",
            "youngMF": "{}".format(ans1[0][0]*0.33),
            "middleAgeMF": "{}".format(ans1[0][1]*0.33),
            "oldMF": "{}".format(ans1[0][2]*0.33),
            "weightMF": "0.33",
            "youngFF": "{}".format(ans2[0][0]*0.33),
            "middleAgeFF": "{}".format(ans2[0][1]*0.33),
            "oldFF": "{}".format(ans2[0][2]*0.33),
            "young": "{}".format(ans[0][0]*0.33 + ans1[0][0]*0.33 + ans2[0][0]*0.33),
            "middleAge": "{}".format(ans[0][1]*0.33 + ans1[0][1]*0.33 + ans2[0][1]*0.33),
            "old": "{}".format(ans[0][2]*0.33 + ans1[0][2]*0.33 + ans2[0][2]*0.33),
            "weightFF": "0.33",
        }
        print(clf1,clf2,clf3)
        print(resultJson)
        return json.dumps(resultJson)

    print(result)
    resultJson = {
        "youngFF": "{}".format(ffMS.model.predict_proba(Xff)[0][0]),
        "middleAgeFF": "{}".format(ffMS.model.predict_proba(Xff)[0][1]),
        "oldFF": "{}".format(ffMS.model.predict_proba(Xff)[0][2]),
        "weightFF": "0.06667",
        "youngMF": "{}".format(mfMS.model.predict_proba(Xmf)[0][0]),
        "middleAgeMF": "{}".format(mfMS.model.predict_proba(Xmf)[0][1]),
        "oldMF": "{}".format(mfMS.model.predict_proba(Xmf)[0][2]),
        "weightMF": "0.06667",
        "youngM": "{}".format(mMS.model.predict_proba(Xm)[0][0]),
        "middleAgeM": "{}".format(mMS.model.predict_proba(Xm)[0][1]),
        "oldM": "{}".format(mMS.model.predict_proba(Xm)[0][2]),
        "weightM": "0.6667",
        "young": "{}".format((result[0][0])),
        "middleAge": "{}".format(result[0][1]),
        "old": "{}".format(result[0][2]),
        "youngAnn": "{}".format(float("{}".format(ANN_Result[0]))),
        "middleAnn": "{}".format(float("{}".format(ANN_Result[1]))),
        "oldAnn": "{}".format(float("{}".format(ANN_Result[2]))),
        "weightAnn": "0.20",
        "error": ""
    }

    return json.dumps(resultJson)


if __name__ == "__main__":  # Main app to run flask on localhost:5000
    app.run()
