import numpy as np
import os
import cv2
from imageio.v2 import imread
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split



from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2


root = 'datasets/Images'
files = os.listdir(root)

images = []
for file in files:
    if file[0] == '.': # to get rid of hidden files
        continue
    dog_type = file.split('-')[1]
    dogs = os.listdir(root +'/' +file)
    dogs = [(root+'/'+file+'/'+dog, dog_type) for dog in dogs if dog[0]!='.']
    images += dogs

dog_df = pd.DataFrame({'image_url': [image[0] for image in images], 'breed': [image[1] for image in images]})
X = [cv2.imread(image[0], cv2.IMREAD_COLOR) for image in images]
del images
dim = (443, 386) # kernel crashed
# resize image
X = [np.array(cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)).astype('float32')/255 for img in X]
y = dog_df['breed']


with open('./datasets/image_data.npy', 'wb') as f:
    np.save(f, X)
    np.save(f, np.array(y))

