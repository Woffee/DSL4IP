#This only work with image and food frame should be bigger than any object around it
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import tensorflow.keras.backend as K
#import shlex e=shelx.split(file_read)
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os

# sys.path.append(BASE_DIR)

class Food():
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        K.clear_session()
        # Load the model
        self.km = load_model(self.BASE_DIR + '/food_detect_model.hdf5',compile = False)
        #Read the calorie and weight dataset
        self.df=pd.read_csv(self.BASE_DIR + '/calorie_data.csv')
        # e is denoted to the list of Food 101 items
        #e=['Apple pie', 'Baby back ribs', 'Baklava', 'Beef carpaccio', 'Beef tartare', 'Beet salad', 'Beignets', 'Bibimbap', 'Bread pudding', 'Breakfast burrito', 'Bruschetta', 'Caesar salad', 'Cannoli', 'Caprese salad', 'Carrot cake', 'Ceviche', 'Cheesecake', 'Cheese plate', 'Chicken curry', 'Chicken quesadilla', 'Chicken wings', 'Chocolate cake', 'Chocolate mousse', 'Churros', 'Clam chowder', 'Club sandwich', 'Crab cakes', 'Creme brulee', 'Croque madame', 'Cup cakes', 'Deviled eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs benedict', 'Escargots', 'Falafel', 'Filet mignon', 'Fish and chips', 'Foie gras', 'French fries', 'French onion soup', 'French toast', 'Fried calamari', 'Fried rice', 'Frozen yogurt', 'Garlic bread', 'Gnocchi', 'Greek salad', 'Grilled cheese sandwich', 'Grilled salmon', 'Guacamole', 'Gyoza', 'Hamburger', 'Hot and sour soup', 'Hot dog', 'Huevos rancheros', 'Hummus', 'Ice cream', 'Lasagna', 'Lobster bisque', 'Lobster roll sandwich', 'Macaroni and cheese', 'Macarons', 'Miso soup', 'Mussels', 'Nachos', 'Omelette', 'Onion rings', 'Oysters', 'Pad thai', 'Paella', 'Pancakes', 'Panna cotta', 'Peking duck', 'Pho', 'Pizza', 'Pork chop', 'Poutine', 'Prime rib', 'Pulled pork sandwich', 'Ramen', 'Ravioli', 'Red velvet cake', 'Risotto', 'Samosa', 'Sashimi', 'Scallops', 'Seaweed salad', 'Shrimp and grits', 'Spaghetti bolognese', 'Spaghetti carbonara', 'Spring rolls', 'Steak', 'Strawberry shortcake', 'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna tartare', 'Waffles']
        self.e=list(self.df['categories'].values)

    def detect(self, img_path):

        im_1=cv2.imread(img_path)
        #remove face from image
        gray = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(self.BASE_DIR + "/haarcascade_frontalface_default.xml")
        fd = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        for (x, y, w, h) in fd:
            cv2.rectangle(im_1, (x, y), (x + w, y + h), (0, 0, 0), 2)
            im_1[y:y+h,x:x+w]=0
        #Convert the image and perform operation so every value of array comes in 0 to 1

        roi=cv2.resize(im_1,(299, 299))
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        roi /= 255.
        #predict the food by the help of model
        pred=self.km.predict(roi)


        score = max(pred[0])
        print("score:", max(pred[0]))

        if score>0.1:
            #print(pred)
            max_e=max(pred)
            ind=pred.argmax()
            # print("ind", ind)
            pred_value=self.e[ind]
            print(pred_value)
        else:
            pred_value = ""
            print("no food")

        #Calculating calories and weight of food
        # print('Calories of food are',end=' ')
        # print(df[df['categories']==pred_value ]['calories'].values[0])

        # print("per weight in grams",end=' ')
        # print(df[df['categories']== pred_value]['weight'].values[0])
        return score, pred_value


if __name__ == '__main__':
    f = Food()
    f.detect("p1.jpg")