import cv2
import dlib
import numpy as np
from imutils import face_utils

FACE_LANDMARKS_DAT_PATH = "C:\\Users\\user\\PycharmProjects\\ImageClassificationMicroservice\\feature_detectors\\shape_predictor_68_face_landmarks.dat"


def mouth_detection(imagePath):
    img = cv2.imread(imagePath)
    points = []
    scale_percent = 300
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("%s" % FACE_LANDMARKS_DAT_PATH)
    faces = detector(gray)
    if len(faces) > 0:
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            landmarks = predictor(image=gray, box=face)

        imgH, imgW, imgC = img.shape
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == "mouth":
                for n in range(i, j):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    points.append([x, y])
        (x, y, w, h) = cv2.boundingRect(np.array([points]))
        roi = img[y:y + h, x:x + w]
        addTop = 30
        addBottom = 10
        addLeft = 10
        addRight = 10
        if x < addLeft:
            addLeft = x
        if (imgW - x) < addRight:
            addRight = imgW - x + (addRight - x)

        roi = img[y - addTop:y + h + addBottom, x - addLeft: x + w + addRight]
        roi = cv2.resize(roi, (300, 170))
        return roi
    return []
