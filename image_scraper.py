import pandas
import requests
import shutil
import os
from multiprocessing.dummy import Pool
from urllib.request import urlretrieve
from os import path

from joblib import Parallel, delayed  
import multiprocessing
import time


def get_image_and_save(i):
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

start = time.time()

fn = '/Users/tonyodongo/Desktop/Herbarium/NEVP_phenology_1.0_scored_20190910.csv'
destination = '/Users/tonyodongo/Desktop/Herbarium/NEVP_images/'
n_images_to_download = 20

data = pandas.read_csv(fn)

if not os.path.exists(destination):
    os.mkdir(destination)

num_cores = multiprocessing.cpu_count()
 
 
 
print("numCores = " + str(num_cores))

Parallel(n_jobs=1)(delayed(get_image_and_save)(i) for i in range(n_images_to_download))

end = time.time()

print(end - start)

