import glob
import pandas as pd
import os
import shutil
import numpy as np

all_filenames = os.listdir("/gpfs/loomis/home.grace/teo22/project/Herbarium/Test_Cropped256_8_8/Not_Flowering/")

filename = '/gpfs/loomis/home.grace/teo22/project/Herbarium/NEVP_out.csv'
data = pd.read_csv(filename)

data_wrong = pd.read_csv('/gpfs/loomis/home.grace/teo22/project/Herbarium/not_flowering_wrong_summary.csv')
filename_wrong = np.unique(data_wrong['Filename'])

fam = []
fam_wrong = []
acc = []
total_count = []

for i in all_filenames:
    selection = pd.Series(data['filename'] == i)
    df = data[selection.values]
    fam_name = np.unique(df['scientificName'])
    fam.append(fam_name)

fam = np.concatenate(fam)
from collections import Counter

families = list(Counter(fam).keys()) # equals to list(set(words))
count = list(Counter(fam).values()) # counts the elements' frequency


for i in filename_wrong:
	selection2 = pd.Series(data['filename'] == i)
	df2 = data[selection2.values]
	fam_name2 = np.unique(df2['scientificName'])
	fam_wrong.append(fam_name2)

fam_wrong = np.concatenate(fam_wrong)

families_wrong = list(Counter(fam_wrong).keys()) # equals to list(set(words))
count_wrong = list(Counter(fam_wrong).values()) # counts the elements' frequency

print(families_wrong)
print(count_wrong)

for i in range(len(families)):
	count_total = count[i]
	
	count_wr = 0
	for j in range(len(families_wrong)):
	#count_wr = 0
		if (families_wrong[j] == families[i]):
			count_wr = count_wrong[j]
			continue
# 	try:
# 		if any((j==families_wrong[i]) for j in families):
# 			count_wr = count_wrong[i]		
# 	except IndexError:
# 		count_wr = 0
		
	accuracy = (count_total - count_wr)/count_total
	acc.append(accuracy)
	total_count.append(count_total)
	
	print (families[i], accuracy)

results = pd.DataFrame({"Family":families, "Accuracy":acc, "Total count": total_count})
                      
results.to_csv("not_flowering_species_accuracy.csv",index=False)












































