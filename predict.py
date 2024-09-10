import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2

# Load the model
age_model = tf.keras.models.load_model('age_model.h5')
gender_model = tf.keras.models.load_model('gender_model.h5')

# Read the image from file
image = cv2.imread('test8.jpg')

# Preprocess the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (128, 128))
image = np.expand_dims(image, axis=0)

# Pre dict the age
age_prediction = age_model.predict(image)[0]
gender_prediction = gender_model.predict(image)[0]

# Get the index of the age with the highest probability
age_index = np.argmax(age_prediction)

# Get the label corresponding to the age index
age_labels = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)', '(15, 23)', '(23, 30)', '(30, 38)', '(43, 48)', '(53, 60)']
age = age_labels[age_index]

if(gender_prediction[1] > 0.5):
    print('Gioi tinh: Nam')
else:
    print('Gioi tinh: Nu')

print(f"Du doan do tuoi: {age}")