# code to get the key frame from the video and save it as a png file.

import sys

import numpy as np

from scipy import spatial

from handshape_feature_extractor import HandShapeFeatureExtractor

from numpy import genfromtxt

import cv2

import os


def get_inference_vector_one_frame_alphabet(files_list):
    model = HandShapeFeatureExtractor.get_instance()

    vectors = []

    video_names = []

    step = int(len(files_list) / 100)

    if step == 0:
        step = 1

    count = 0

    for video_frame in files_list:

        img = cv2.imread(video_frame)

        #img = cv2.rotate(img, cv2.ROTATE_180)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results = model.extract_feature(img)

        results = np.squeeze(results)

        vectors.append(results)

        video_names.append(os.path.basename(video_frame))

        count += 1

        if count % step == 0:
            sys.stdout.write("_")

            sys.stdout.flush()

    return vectors


# videopath : path of the video file

# frames_path: path of the directory to which the frames are saved

# count: to assign the video order to the frane.

def frameExtractor(videopath, frames_path, count):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)

    cap = cv2.VideoCapture(videopath)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    frame_no = int(video_length / 1.3)

    # print("Extracting frame..\n")

    cap.set(1, frame_no)

    ret, frame = cap.read()

    cv2.imwrite(frames_path + "/%#05d.png" % (count + 1), frame)

