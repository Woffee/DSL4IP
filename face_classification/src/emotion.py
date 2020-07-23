"""


@Time    : 7/19/20
@Author  : Wenbo
"""

import sys
from PIL import Image, ImageDraw, ImageFont
import cv2
from keras.models import load_model
import numpy as np
from PyQt5.QtGui import QFont, QPixmap

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

# parameters for loading data and images
# image_path = sys.argv[1]
# image_path = '/Users/woffee/Pictures/facial_expression2_040414.jpg'


class Emotion():
    def __init__(self):
        detection_model_path = BASE_DIR + '/../trained_models/detection_models/haarcascade_frontalface_default.xml'
        emotion_model_path = BASE_DIR + '/../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
        gender_model_path = BASE_DIR + '/../trained_models/gender_models/simple_CNN.81-0.96.hdf5'

        self.emotion_labels = get_labels('fer2013')
        self.gender_labels = get_labels('imdb')
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # loading models
        self.face_detection = load_detection_model(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.gender_classifier = load_model(gender_model_path, compile=False)

        # getting input model shapes for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.gender_target_size = self.gender_classifier.input_shape[1:3]

        # hyper-parameters for bounding boxes shape
        self.gender_offsets = (30, 60)
        self.gender_offsets = (10, 10)
        self.emotion_offsets = (20, 40)
        self.emotion_offsets = (0, 0)


    def detect(self, image_path):
        # loading images
        rgb_image = load_image(image_path, grayscale=False)
        gray_image = load_image(image_path, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')

        res = []

        faces = detect_faces(self.face_detection, gray_image)
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, self.gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]

            x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                rgb_face = cv2.resize(rgb_face, (self.gender_target_size))
                gray_face = cv2.resize(gray_face, (self.emotion_target_size))
            except:
                continue

            rgb_face = preprocess_input(rgb_face, False)
            rgb_face = np.expand_dims(rgb_face, 0)
            gender_prediction = self.gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = self.gender_labels[gender_label_arg]

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(self.emotion_classifier.predict(gray_face))
            emotion_text = self.emotion_labels[emotion_label_arg]

            if gender_text == self.gender_labels[0]:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            res.append({
                'left': x1,
                'top': y1,
                'right': x2,
                'bottom': y2,
                'male': gender_text,
                'emotion': emotion_text
            })

            # print(gender_text, emotion_text)
            # draw = ImageDraw.Draw(img)
            # draw.rectangle(((x1, y1), (x2,y2)), outline='red')
            # draw.text((0, 0), "something123")
        return res