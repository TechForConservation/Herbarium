import glob
import pandas as pd
import os
import shutil
import numpy as np

#all_filenames = os.listdir("/gpfs/loomis/home.grace/teo22/project/Herbarium/Test_Resized224/Flowering/")

#flowering_dir = '/gpfs/loomis/home.grace/teo22/project/Herbarium/Test_Resized224/Flowering/'
#not_flowering_dir = '/gpfs/loomis/home.grace/dollar/teo22/project/Herbarium/Train/Not_Flowering/'
#filename = './NEVP_out.csv'
filename = '/gpfs/loomis/home.grace/teo22/project/Herbarium/NEVP_out.csv'
data = pd.read_csv(filename)

data_right = pd.read_csv('/gpfs/loomis/home.grace/teo22/project/Herbarium/flowering_right_summary.csv')
filename_right = np.unique(data_right['Filename'])

data_wrong = pd.read_csv('/gpfs/loomis/home.grace/teo22/project/Herbarium/flowering_wrong_summary.csv')
filename_wrong = np.unique(data_wrong['Filename'])

#fam_names = np.unique(data['family'])
fam = []
fam_wrong = []
acc = []
total_count = []

for i in filename_right:
    selection = pd.Series(data['filename'] == i)
    df = data[selection.values]
    fam_name = np.unique(df['family'])
    fam.append(fam_name)
    #print(fam)

#fam = fam.tolist()
fam = np.concatenate(fam)
from collections import Counter

families = list(Counter(fam).keys()) # equals to list(set(words))
count = list(Counter(fam).values()) # counts the elements' frequency

#print(families[3])
#print(count[0])

for i in filename_wrong:
    selection = pd.Series(data['filename'] == i)
    df = data[selection.values]
    fam_name = np.unique(df['family'])
    fam_wrong.append(fam_name)
    #print(fam)

#fam = fam.tolist()
fam_wrong = np.concatenate(fam_wrong)

families_wrong = list(Counter(fam_wrong).keys()) # equals to list(set(words))
count_wrong = list(Counter(fam_wrong).values()) # counts the elements' frequency

#print(families_wrong)
#print(count_wrong)

for i in range(len(families)):
	count_right = count[i]
	
	for j in range(len(families_wrong)):
		if any((k==families_wrong[j]) for k in families):
			count_wr = count_wrong[j]
		
	accuracy = count_right/(count_right+count_wr)
	acc.append(accuracy)
	total_count.append(count_right+count_wr)
	
	print (families[i], accuracy)

results = pd.DataFrame({"Family":families, "Accuracy":acc, "Total count": total_count})
                      
results.to_csv("flowering_family_accuracy2.csv",index=False)












































