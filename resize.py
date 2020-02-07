import os, sys
from PIL import Image


path = ('/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/Train_Cropped/Flowering/')
new_path = ('/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/Train_Cropped224/Flowering/')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize():
	print(path)
	for item in os.listdir(path):
		#print(item)
		file_path = os.path.join(path, item)
		if os.path.isfile(file_path):
			print(file_path)
			im = Image.open(file_path)
			f, e = os.path.splitext(file_path)
			file_name = f.split('/')[-1]
			imResize = im.resize((224,224), Image.ANTIALIAS)
			imResize.save(new_path + file_name + '.jpg', 'JPEG')
resize()