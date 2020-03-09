import glob
import pandas as pd
import os
import shutil
import numpy as np

from itertools import product

comb = list(product([0.0,1.0],repeat=12))
#print(comb[1])

filename = '/Users/tonyodongo/Desktop/not_flowering_right_summary.csv'
filename2 = '/Users/tonyodongo/Desktop/not_flowering_wrong_summary.csv'
data = pd.read_csv(filename)
data2 = pd.read_csv(filename2)

filenames = data['Filename']
flowering = data['Flowering']
budding = data['Budding']
fruiting = data['Fruiting']
mostly_buds = data['Mostly buds']
mostly_mature = data['Mostly mature']
mostly_old = data['Mostly old']
mostly_open = data['Mostly open']
mostly_young = data['Mostly young']
not_scorable = data['Not scorable']
past_maturity = data['Past maturity']
reproductive = data['Reproductive']
sterile = data['Sterile']

filenames2 = data2['Filename']
flowering2 = data2['Flowering']
budding2 = data2['Budding']
fruiting2 = data2['Fruiting']
mostly_buds2 = data2['Mostly buds']
mostly_mature2 = data2['Mostly mature']
mostly_old2 = data2['Mostly old']
mostly_open2 = data2['Mostly open']
mostly_young2 = data2['Mostly young']
not_scorable2 = data2['Not scorable']
past_maturity2 = data2['Past maturity']
reproductive2 = data2['Reproductive']
sterile2 = data2['Sterile']



# (budding, fruiting, mostly_buds, mostly_mature, 
# mostly_old, mostly_open, mostly_young, not_scorable, 
# past_maturity, reproductive, sterile) = [np.zeros((filenames.size)) for _ in range(11)]

#print(len(filenames))
# for j in range(filenames.size):
#print([budding[j], fruiting[j], mostly_buds[j], mostly_mature[j], mostly_old[j], mostly_open[j], mostly_young[j], not_scorable[j], past_maturity[j], reproductive[j], sterile[j]])

#print((budding[1], fruiting[1], mostly_buds[1], mostly_mature[1], mostly_old[1], mostly_open[1], mostly_young[1], not_scorable[1], past_maturity[1], reproductive[1], sterile[1]))

#count = 0
#for j in range(filenames.size):

comb_count = 0

clist = []
acc = []
count = []

for c in comb:
	count_right = 0
	count_wrong = 0

	for j in range(filenames.size):
		#print(type(c))
		#count = 0
		if (flowering[j], budding[j], fruiting[j], mostly_buds[j], mostly_mature[j], mostly_old[j], mostly_open[j], mostly_young[j], not_scorable[j], past_maturity[j], reproductive[j], sterile[j]) == c:
			count_right = count_right + 1
			continue
	
	for j in range(filenames2.size):
		#print(type(c))
		#count = 0
		if (flowering2[j], budding2[j], fruiting2[j], mostly_buds2[j], mostly_mature2[j], mostly_old2[j], mostly_open2[j], mostly_young2[j], not_scorable2[j], past_maturity2[j], reproductive2[j], sterile2[j]) == c:
			count_wrong = count_wrong + 1
			continue
	
	if (count_right + count_wrong) > 0:
		accuracy = count_right/(count_right+count_wrong)
		#print ((budding[j], fruiting[j], mostly_buds[j], mostly_mature[j], mostly_old[j], mostly_open[j], mostly_young[j], not_scorable[j], past_maturity[j], reproductive[j], sterile[j]), accuracy)
		
		print(c, accuracy)
		
		comb_count = comb_count + 1;

		clist.append(list(c))
		acc.append(accuracy)
		count.append(count_right+count_wrong)


(flowering3, budding3, fruiting3, mostly_buds3, mostly_mature3, 
mostly_old3, mostly_open3, mostly_young3, not_scorable3, 
past_maturity3, reproductive3, sterile3, accuracy3, count3) = [np.zeros(comb_count) for _ in range(14)]

		
for i in range(comb_count):
	flowering3[i] = clist[i][0]
	budding3[i] = clist[i][1]
	fruiting3[i] = clist[i][2]
	mostly_buds3[i] = clist[i][3]
	mostly_mature3[i] = clist[i][4]
	mostly_old3[i] = clist[i][5]
	mostly_open3[i] = clist[i][6]
	mostly_young3[i] = clist[i][7]
	not_scorable3[i] = clist[i][8]
	past_maturity3[i] = clist[i][9]
	reproductive3[i] = clist[i][10]
	sterile3[i] = clist[i][11]
	accuracy3[i] = acc[i]
	count3[i] = count[i]
		
results = pd.DataFrame({"Flowering":flowering3,
                      "Budding":budding3, "Fruiting": fruiting3, 
                      "Mostly buds":mostly_buds3, "Mostly mature": mostly_mature3, "Mostly old": mostly_old3,
                      "Mostly open":mostly_open3, "Mostly young": mostly_young3, "Not scorable": not_scorable3,
                      "Past maturity":past_maturity3, "Reproductive": reproductive3, "Sterile": sterile3, "Accuracy": accuracy3, "Count": count3})
                      
results.to_csv("not_flowering_accuracy.csv",index=False)




















































