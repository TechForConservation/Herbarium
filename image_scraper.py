import pandas 
import requests
import shutil
import os
from os import path

fn = '/Users/zachcm/Downloads/NEVP_phenology_1.0_scored_20190910.csv'
destination = '/Users/zachcm/Downloads/NEVP_images/'
n_images_to_download = 100

data = pandas.DataFrame.from_csv(fn)

if not os.path.exists(destination):
    os.mkdir(destination)

for url in data['originalurl'][0:n_images_to_download]:
    print(url)
    r = requests.get(url, stream=True)
    path = "{}{}.jpg".format(destination, url.split('/')[-1])
    with open(path, 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)