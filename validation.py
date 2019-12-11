import glob
import pandas
import os
import shutil
import random
import numpy as np

validation_dir = './Valid/'
images = glob.glob("./NEVP_images/*.jpg")
i = np.random.choice(len(images),int(0.1*len(images)),replace=False)

if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
   
for im in i:
    shutil.move(images[im], validation_dir)