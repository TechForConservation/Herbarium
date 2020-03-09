import glob
import pandas as pd
import os
import shutil
import numpy as np

#all_filenames = os.listdir("/gpfs/loomis/home.grace/teo22/project/Herbarium/Test_Cropped256_8_8/Flowering/")

filename = './results_xception_83.669%_confidence_train_set.csv'
data = pd.read_csv(filename)

#data_wrong = pd.read_csv('/gpfs/loomis/home.grace/teo22/project/Herbarium/flowering_wrong_summary.csv')
#filename_wrong = np.unique(data_wrong['Filename'])

filtered_name = []
filtered_name_not = []
predictions = []
predictions_not = []

# filtered_name_r = []
# filtered_name_not_r = []
# predictions_r = []
# predictions_not_r = []
# 
# filtered_name_w = []
# filtered_name_not_w = []
# predictions_w = []
# predictions_not_w = []
# fam_wrong = []
# acc = []
# total_count = []

#print(data['Filename'][0][0])

for i in range(len(data)):
	filename = data['Filename'][i]
	filter = filename.split('/')[1]
	if filename[0] == 'F':
		filtered_name.append(filter)
		predictions.append(data['Predictions'][i])
	else:
		filtered_name_not.append(filter)
		predictions_not.append(data['Predictions'][i])

# for i in range(len(filtered_name)):
# 	filename = filtered_name[i]
# 	if predictions[i] == 'Flowering':
#     	filtered_name_r.append(filename)
#     	predictions_r.append(predictions[i])
# 	else
#     	filtered_name_w.append(filename)
#     	predictions_w.append(predictions[i])		
# 
# for i in range(len(filtered_name_not)):
# 	filename = filtered_name_not[i]
# 	if predictions_not[i] == 'Not_Flowering':
#     	filtered_name_not_r.append(filename)
#     	predictions_not_r.append(predictions_not[i])
# 	else
#     	filtered_name_not_w.append(filename)
#     	predictions_not_w.append(predictions_not[i])

results_flowering = pd.DataFrame({"filtered_name":filtered_name, "predictions":predictions})                     
results_flowering.to_csv("flowering_train_set_predictions.csv",index=False)

results_not_flowering = pd.DataFrame({"filtered_name":filtered_name_not, "predictions":predictions_not})                     
results_not_flowering.to_csv("not_flowering_train_set_predictions.csv",index=False)









































