import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from tensorflow.keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from IPython.display import SVG
#from tensorflow.keras.utils.vis_utils import model_to_dot

import time
import numpy as np

FAST_RUN = False
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
NAME = "flowering-vs-non_flowering-densenet-224-cropped-2-{}".format(int(time.time()))

def densenet(img_shape, n_classes, f=32):
  repetitions = 6, 12, 24, 16
  
  def bn_rl_conv(x, f, k=1, s=1, p='same'):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(f, k, strides=s, padding=p)(x)
    return x
  
  
  def dense_block(tensor, r):
    for _ in range(r):
      x = bn_rl_conv(tensor, 4*f)
      x = bn_rl_conv(x, f, 3)
      tensor = Concatenate()([tensor, x])
    return tensor
  
  
  def transition_block(x):
    x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
    x = AvgPool2D(2, strides=2, padding='same')(x)
    return x
  
  
  input = Input(img_shape)
  
  x = Conv2D(64, 7, strides=2, padding='same')(input)
  x = MaxPool2D(3, strides=2, padding='same')(x)
  
  for r in repetitions:
    d = dense_block(x, r)
    x = transition_block(d)
  
  x = GlobalAvgPool2D()(d)
  
  output = Dense(n_classes, activation='softmax')(x)
  
  model = Model(input, output)
  return model
  
img_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
n_classes = 2

model = densenet(img_shape, n_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

earlystop = EarlyStopping(patience=20)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
#                                             patience=2, 
#                                             verbose=1, 
#                                             factor=0.5, 
#                                             min_lr=0.00001)

filepath="flowering_densenet_224_cropped_2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks = [earlystop, tensorboard, checkpoint]

nb_train_samples = 34582
nb_validation_samples = 8645
batch_size=16

train_path = '/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/Train_Cropped224_2'
valid_path = '/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/Valid_Cropped224_2'

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
    callbacks=callbacks,
    use_multiprocessing=True,
    workers=4
)

  
  
  
  
  
  
  
  





















