import glob
import pandas as pd
import os
import shutil
import numpy as np

filename = './flowering_right_train_set_predictions_reduced.csv'
data = pd.read_csv(filename)


filenames = np.unique(data['filename'])


# (budding, flowering, fruiting, mostly_buds, mostly_mature, 
# mostly_old, mostly_open, mostly_young, not_scorable, 
# past_maturity, reproductive, sterile) = [np.zeros((filenames.size)) for _ in range(12)]

(budding, flowering, fruiting, mostly_buds, mostly_mature, 
mostly_old, mostly_open, mostly_young, not_scorable, 
past_maturity, reproductive, sterile) = [np.zeros((filenames.size)) for _ in range(12)]

print(len(filenames))

for i in range(filenames.size):
	selection = pd.Series(data['filename'] == filenames[i])
	df = data[selection.values]
	
	if any(df['stateName'] == 'Budding'):
		budding[i] = 1
	else:
		budding[i] = 0
		
	if any(df['stateName'] == 'Flowering'):
		flowering[i] = 1
	else:
		flowering[i] = 0

	if any(df['stateName'] == 'Fruiting'):
		fruiting[i] = 1
	else:
		fruiting[i] = 0
		
	if any(df['stateName'] == 'Mostly buds'):
		mostly_buds[i] = 1
	else:
		mostly_buds[i] = 0

	if any(df['stateName'] == 'Mostly masture'):
		mostly_mature[i] = 1
	else:
		mostly_mature[i] = 0
		
	if any(df['stateName'] == 'Mostly old'):
		mostly_old[i] = 1
	else:
		mostly_old[i] = 0

	if any(df['stateName'] == 'Mostly open'):
		mostly_open[i] = 1
	else:
		mostly_open[i] = 0
		
	if any(df['stateName'] == 'Mostly young'):
		mostly_young[i] = 1
	else:
		mostly_young[i] = 0

	if any(df['stateName'] == 'Not scorable'):
		not_scorable[i] = 1
	else:
		not_scorable[i] = 0
		
	if any(df['stateName'] == 'Past maturity'):
		past_maturity[i] = 1
	else:
		past_maturity[i] = 0

	if any(df['stateName'] == 'Reproductive'):
		reproductive[i] = 1
	else:
		reproductive[i] = 0
		
	if any(df['stateName'] == 'Sterile'):
		sterile[i] = 1
	else:
		sterile[i] = 0

# results = pd.DataFrame({"Filename":filenames,
#                       "Budding":budding, "Flowering": flowering, "Fruiting": fruiting, 
#                       "Mostly buds":mostly_buds, "Mostly mature": mostly_mature, "Mostly old": mostly_old,
#                       "Mostly open":mostly_open, "Mostly young": mostly_young, "Not scorable": not_scorable,
#                       "Past maturity":past_maturity, "Reproductive": reproductive, "Sterile": sterile})

results = pd.DataFrame({"Filename":filenames, "Flowering":flowering,
                      "Budding":budding, "Fruiting": fruiting, 
                      "Mostly buds":mostly_buds, "Mostly mature": mostly_mature, "Mostly old": mostly_old,
                      "Mostly open":mostly_open, "Mostly young": mostly_young, "Not scorable": not_scorable,
                      "Past maturity":past_maturity, "Reproductive": reproductive, "Sterile": sterile})

results.to_csv("flowering_right_train_set_summary.csv",index=False)



















































