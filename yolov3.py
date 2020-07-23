"""


@Time    : 7/18/20
@Author  : Wenbo
"""
from darknetpy.detector import Detector


class Yolo():
    def __init__(self):
        abs_path = '/Users/woffee/www/language_design/yolo/'
        self.detector = Detector(abs_path + 'darknet/cfg/coco.data',
                            abs_path + 'darknet/cfg/yolov3.cfg',
                            abs_path + 'darknet/yolov3.weights')

    def detect(self, img_path):
        results = self.detector.detect(img_path)
        return results


if __name__ == '__main__':
    y = Yolo()
    r = y.detect('/Users/woffee/www/language_design/couples-posing-800x450.jpg')
    print(r)
