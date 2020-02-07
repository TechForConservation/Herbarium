import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import time
#print(os.listdir("../input"))

# import tensorflow.keras.backend as K
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
# from tensorflow.keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
# from tensorflow.keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU
# #from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# 
# from IPython.display import SVG
# #from tensorflow.keras.utils.vis_utils import model_to_dot

FAST_RUN = False
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
NAME = "flowering-vs-non_flowering-densenet-224-cropped-cont2-{}".format(int(time.time()))
# 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.models import load_model

# model = Sequential()
# 
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# 
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# 
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# 
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# 
# model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes
# 
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model = load_model('flowering_densenet_224_cropped_2_82.257%.hdf5')

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

earlystop = EarlyStopping(patience=20)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
#                                             patience=2, 
#                                             verbose=1, 
#                                             factor=0.5, 
#                                             min_lr=0.00001)

filepath="flowering_densenet_224_cropped_cont2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks = [earlystop, tensorboard, checkpoint]

nb_train_samples = 34582
nb_validation_samples = 8645
batch_size=16

train_path = '/gpfs/loomis/home.grace/teo22/project/Herbarium/Train_Cropped224_2'
valid_path = '/gpfs/loomis/home.grace/teo22/project/Herbarium/Valid_Cropped224_2'

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_generator = train_datagen.flow_from_directory(
    train_path, target_size=IMAGE_SIZE, class_mode='categorical', classes=['Flowering', 'Not_Flowering'], batch_size=batch_size)
    
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    valid_path, target_size=IMAGE_SIZE, class_mode='categorical', classes=['Flowering', 'Not_Flowering'], batch_size=batch_size)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
    
epochs=3 if FAST_RUN else 500
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    steps_per_epoch=nb_train_samples//batch_size,
    callbacks=callbacks
)
