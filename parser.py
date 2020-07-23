from lark.lark import Lark
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math

from functools import partial
from img_modifier import img_helper
from img_modifier import color_filter
from PIL import ImageQt

from yolov3 import Yolo
from agender import Agender
from face_classification.src.emotion import Emotion
from food_detection.food_detection import Food

ipl_grammar = """
    start: statement+

    statement:  NUMBER                                           -> number
               | TUNEOP + "the" + ATTRIBUTE + "by" + NUMBER      -> tuning
               | "apply a filter" +  FILTER                      -> set_filter
               | "flip the image" + FLIPDIR                      -> flip
               | "detect this image"                             -> detect
               | "how many" + OBJECTS + "in this image"          -> how_many
               | "tag the" + OBJECTS + "in this image"           -> tag_objects
               | "what are in this image?"                       -> show_statistics
               | "detect food in this image"                     -> detect_food
               

    MATHOP: "+"|"-"|"*"|"/"
    IMAGE: (LETTER|NUMBER|"_")+"."LETTER+
    IDENTIFIER: LETTER(LETTER|NUMBER)*
    TUNEOP: "increase" | "decrease"
    ATTRIBUTE: "brightness" | "contrast" | "sharpness"
    FILTER: "sepia" | "negative" | "black_white"
    FLIPDIR: "horizontally" | "vertically"
    OBJECTS: LETTER+
    

    %import common.LETTER
    %import common.INT -> NUMBER
    %import common.WS
    %ignore WS
"""


class IPDSL():
    def __init__(self):
        self.parser = Lark(ipl_grammar)
        self.yolo = None
        self.agender = None
        self.emotion = None
        self.food = None

        self.original_img = None
        self.img_path = ''
        self.msg = ''
        self.persons = []
        self.objects = []
        pass

    def interpreter(self, s, img, img_path = ''):
        if s.data == 'tuning':
            tmp = 1 if s.children[0] == 'increase' else -1
            attr = s.children[1]
            val = int(s.children[2])

            print(tmp, attr, val)
        elif s.data == 'set_filter':
            filter_name = s.children[0]
            return img_helper.color_filter(img, filter_name)

        elif s.data == 'flip':
            flip_dir = s.children[0]
            if flip_dir == 'vertically':
                return img_helper.flip_top(img)
            elif flip_dir == 'horizontally':
                return img_helper.flip_left(img)

        elif s.data == 'detect':
            print(img_path)
            if self.yolo is None:
                self.yolo = Yolo()
                self.agender = Agender()
                self.emotion = Emotion()

            if img_path != self.img_path:
                self.img_path = img_path
                self.persons = []
                self.objects = []

            objects = self.yolo.detect(img_path)
            faces = self.agender.detect(img_path)
            emotions = self.emotion.detect(img_path)

            font_path = "/Users/woffee/www/language_design/prototype/font_consolas/CONSOLA.TTF"
            font = ImageFont.truetype(font_path, 20)
            fw = 11
            fh = 16

            msg = []
            draw = ImageDraw.Draw(img)

            for i, box in enumerate(objects):
                l = box['left']
                t = box['top']
                b = box['bottom']
                r = box['right']
                label = box['class']

                self.objects.append({
                    'left': l,
                    'top': t,
                    'bottom': b,
                    'right': r,
                    'class': label
                })

                if label == 'person':
                    self.persons.append({
                        'left': l,
                        'top': t,
                        'bottom': b,
                        'right': r,
                        'gender': '',
                        'age':0,
                        'emotion':''
                    })
                else:
                    draw.rectangle(((l, t), (r, b)), outline='blue')
                    txt_width = fw * len(label)
                    draw.rectangle(((l, t), (l + txt_width, t + fh)), fill="blue")
                    draw.text((l, t), label, font=font)

            for i, box in enumerate(faces):
                l = box['left']
                t = box['top']
                b = box['bottom']
                r = box['right']
                gender = 'male' if box['gender'] <0.5 else 'female'
                age = box['age']
                label = gender + ", %.2f" % age
                print(" * Agender: " + label)
                # msg.append(" * Agender: " + label)

                score = 0
                for i, p in enumerate(self.persons):
                    area = self.computeArea(l, t, r, b, p['left'], p['top'], p['right'], p['bottom'])
                    s = area / ( (r-l) * (b-t) )
                    if s > 0.5:
                        self.persons[i]['age'] = age
                        self.persons[i]['gender'] = gender

            for i, box in enumerate(emotions):
                l = box['left']
                t = box['top']
                b = box['bottom']
                r = box['right']
                emo = box['emotion']

                print(" * Emotion: " + emo)
                # msg.append(" * Emotion: " + emo)

                for i, p in enumerate(self.persons):
                    area = self.computeArea(l, t, r, b, p['left'], p['top'], p['right'], p['bottom'])
                    if (r-l) * (b-t) > 0:
                        s = area / ( (r-l) * (b-t) )
                        if s > 0.5:
                            self.persons[i]['emotion'] = emo

                # draw.rectangle(((l, t), (r, b)), outline='yellow')
                # txt_width = fw * len(emo)
                # draw.rectangle(((l, t), (l + txt_width, t + fh)), fill="black")
                # draw.text((l, t), emo, font=font)

            for i, box in enumerate(self.persons):
                l = box['left']
                t = box['top']
                b = box['bottom']
                r = box['right']
                draw.rectangle(((l, t), (r, b)), outline='blue')

                gender_age = box['gender']
                if box['age'] > 0:
                    gender_age = gender_age + ", age %.2f" % box['age']
                emotion = box['emotion']

                txt_width = fw * len(gender_age)
                draw.rectangle(((l, t), (l + txt_width, t + fh)), fill="blue")
                draw.text((l, t), gender_age, font=font)

                txt_width = fw * len(emotion)
                draw.rectangle(((l, t + fh), (l + txt_width, t + fh * 2)), fill="blue")
                draw.text((l, t + fh), emotion, font=font)

            self.msg = " * done"

        elif s.data == 'how_many':
            str = s.children[0]
            str = self.remove_s(str)
            num = 0
            for obj in self.objects:
                if obj['class'] == str:
                    num += 1
            self.msg = " * %d %s(s)" % (num, str)

        elif s.data == 'tag_objects':
            img = self.original_img.copy()

            str = s.children[0]
            str = self.remove_s(str)

            msg = []
            draw = ImageDraw.Draw(img)

            font_path = "/Users/woffee/www/language_design/prototype/font_consolas/CONSOLA.TTF"
            font = ImageFont.truetype(font_path, 20)
            fw = 11
            fh = 16

            for i, box in enumerate(self.objects):
                if str == box['class']:
                    l = box['left']
                    t = box['top']
                    b = box['bottom']
                    r = box['right']
                    label = box['class']
                    print(" * Yolo: " + label)
                    msg.append(" * Yolo: " + label)

                    draw.rectangle(((l, t), (r, b)), outline='blue')
                    txt_width = fw * len(label)
                    draw.rectangle(((l, t), (l + txt_width, t + fh)), fill="blue")
                    draw.text((l, t), label, font=font)

            self.msg = "\n".join(msg)

        elif s.data == 'show_statistics':
            statistics = {}
            for o in self.objects:
                if o['class'] not in statistics.keys():
                    statistics[o['class']] = 1
                else:
                    statistics[o['class']] += 1
            msg = []
            for k in statistics.keys():
                v = statistics[k]
                msg.append( " * %d %s(s)" % (v, k) )
            self.msg = "\n".join(msg)

        elif s.data == 'detect_food':
            self.detect_food(img_path)

        return img

    def process(self, img, cmd, img_path):
        ast = self.parser.parse(cmd)
        print(ast.pretty())

        for s in ast.children:
            img = self.interpreter(s, img, img_path)
        return img

    def computeArea(self, A, B, C, D, E, F, G, H):
        width = min(C, G) - max(A, E)
        height = min(D, H) - max(B, F)
        # width = width if width > 0 else 0
        # height = height if height > 0 else 0
        # return (C-A)*(D-B) + (G-E)*(H-F) - width*height
        return width * height

    def remove_s(self, str):
        dic = {
            'persons': 'person',
            'dogs': 'dog',
            'cats': 'cat',
            'bicycles': 'bicycle',
            'birds': 'bird',
            'boats': 'boat',
            'cars': 'car',
            'chairs': 'chair',
            'cows': 'cow',
            'diningtables': 'diningtable',
            'horses': 'horse',
            'motorbikes': 'motorbike',
            'pottedplants': 'pottedplant',
            'sheep': 'sheep',
            'sofas': 'sofa',
            'trains': 'train',
            'tvmonitors': 'tvmonitor',
            'trucks': 'truck',
        }
        if str in dic.keys():
            str = dic[str]
        return str

    def detect_food(self, img_path):
        if self.food is None:
            self.food = Food()
        score, name = self.food.detect(img_path)
        if score > 0.1:
            self.msg = " * " + name
        else:
            self.msg = " * no food"


if __name__ == '__main__':
    p = IPDSL()
    # p.process('increase the brightness by 20')
    # p.process('apply a filter black_white')