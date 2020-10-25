import os
import sys
import cv2
from skimage.color import label2rgb
import numpy as np
from feature_detectors.mouthDetection import mouth_detection
from skimage import feature
import matplotlib.pyplot as plt
from PIL import Image
from imutils import face_utils
from math import atan2, degrees
from collections import OrderedDict
from timeit import default_timer as timer


# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def home():
#    return 'Model Entry Point'


# if __name__ == '__main__':
#    app.run()

def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


radius = 3
n_points = 24
path = './photos/pexels-photo-614810.jpeg'
image = cv2.imread(path)
window_name = 'image'



roi = mouth_detection(path)
cv2.imshow(window_name, roi)
cv2.waitKey(0)

"""
face = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
hist_face = feature.local_binary_pattern(face, n_points, radius, "uniform")
fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
plt.gray()

titles = ('edge', 'flat', 'corner')
w = width = radius - 1
edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
i_14 = n_points // 4  # 1/4th of the histogram
i_34 = 3 * (n_points // 4)  # 3/4th of the histogram
corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                 list(range(i_34 - w, i_34 + w + 1)))

label_sets = (edge_labels, flat_labels, corner_labels)

for ax, labels in zip(ax_img, label_sets):
    ax.imshow(overlay_labels(face, hist_face, labels))

for ax, labels, name in zip(ax_hist, label_sets, titles):
    counts, _, bars = hist(ax, hist_face)
    highlight_bars(bars, labels)
    ax.set_ylim(top=np.max(counts[:-1]))
    ax.set_xlim(right=n_points + 2)
    ax.set_title(name)

ax_hist[0].set_ylabel('Percentage')
for ax in ax_img:
    ax.axis('off')
    
"""
