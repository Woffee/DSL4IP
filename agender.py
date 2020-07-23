"""


@Time    : 7/18/20
@Author  : Wenbo
"""
from pyagender import PyAgender
import cv2

class Agender():
    def __init__(self):
        self.agender = PyAgender()

    def detect(self, img_path):
        # img_path = '/Users/woffee/www/language_design/couples-posing-800x450.jpg'
        faces = self.agender.detect_genders_ages(cv2.imread(img_path))
        return faces



