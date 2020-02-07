import os, sys
from PIL import Image

path = ('/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/Train/Flowering/')
new_path = ('/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/Train_Cropped/Flowering/')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def crop():
	print(path)
	for item in os.listdir(path):
		#print(item)
		file_path = os.path.join(path, item)
		if os.path.isfile(file_path):
			print(file_path)
			original = Image.open(file_path)
			
			width, height = original.size   # Get dimensions
			left = width/4
			top = height/4
			right = 3 * width/4
			bottom = 3 * height/4
			#cropped_example = original.crop((left, top, right, bottom))

			f, e = os.path.splitext(file_path)
			file_name = f.split('/')[-1]
			imCrop = original.crop((left, top, right, bottom)), Image.ANTIALIAS)
			imCrop.save(new_path + file_name + '.jpg', 'JPEG')
crop()

#test_image = "1dd17431285aacd6e1ba57563cc43c720acfe5c6b471f3f2a4917f46764d0d7c.jpg"
#original = Image.open(test_image)
#original.show()

# width, height = original.size   # Get dimensions
# left = width/4
# top = height/4
# right = 3 * width/4
# bottom = 3 * height/4
# cropped_example = original.crop((left, top, right, bottom))

#cropped_example.show()