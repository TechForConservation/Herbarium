import glob
import pandas
import os
import shutil

flowering_dir = '/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/Train/Flowering/'
not_flowering_dir = '/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/Train/Not_Flowering/'
#filename = './NEVP_out.csv'
filename = '/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/NEVP_out.csv'
data = pandas.read_csv(filename)

#images = glob.glob("/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/NEVP_images/*.jpg")
#python /gpfs/loomis/home.grace/dollar/teo22/project/

images = glob.glob("./NEVP_images/*.jpg")

if not os.path.exists(flowering_dir):
    os.mkdir(flowering_dir)
if not os.path.exists(not_flowering_dir):
    os.mkdir(not_flowering_dir)
   
for im in images:
    selection = pandas.Series(data['filename'] == im.split('/')[2])
    df = data[selection.values]
   
    if any(df['stateName'] == 'Flowering'):
        shutil.move(im, flowering_dir)
        print('Flowering')
    else:
        shutil.move(im, not_flowering_dir)
        print('Not Flowering')