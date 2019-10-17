import pandas
import requests
import shutil
import os
from os import path

fn = '/Users/tonyodongo/Desktop/Herbarium/NEVP_phenology_1.0_scored_20190910.csv'
destination = '/Users/tonyodongo/Desktop/Herbarium/NEVP_images/'
n_images_to_download = 10

data = pandas.read_csv(fn)

if not os.path.exists(destination):
    os.mkdir(destination)

for i in range(n_images_to_download):
    url = data['originalurl'][i]
    print(url)
    r = requests.get(url, stream=True)
    state = data['stateName'][i]
    family_name = data['family'][i]
    species_name = data['scientificName'][i]
    if not os.path.exists(destination + state + '/'):
        os.mkdir(destination + state + '/')
    path = "{}.jpg".format(destination + state + '/' + family_name + species_name)
    with open(path, 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)