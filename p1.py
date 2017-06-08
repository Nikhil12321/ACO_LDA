import csv
from random import randint
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
import get_mi


class feature(object):

	def __init__(self):
		self.delta = 0
		self.pheromone = 0


class ant(object):

	def __init__(self):
		self.usm = []
		self.ant_f = []
		self.mse = 0

def getData():

	temp_list = []
	global num_tuples
	global num_features

	with open(filename, 'r') as c:
		reader = csv.reader(c)

		for row in reader:
	
			ans = row.pop()
			num_features = len(row)
			if(len(data) == 0):
				f_ans = ans
			if(ans == f_ans):
				result_data.append(0)
			else:
				result_data.append(1)

			t = []

			#convert from string to float
			for v in row:
				t.append(float(v))
			data.append(t)
	num_tuples = len(data)
		
def initFeatures():
	for i in range(0, num_features):
		temp = feature()
		temp.delta = 0
		temp.pheromone = cc
		features.append(temp)

def initAnt(ants, real_ants):

	global p

	a = ant()
	temp_selected_features = list(selected_features)

	for i in range(0, m-p):

		r = randint(0, len(temp_selected_features)-1)
		a.ant_f.append(temp_selected_features[r])
		temp_selected_features.remove(temp_selected_features[r])


	#Apply USM and other measures here for the next p num_features
	for i in range(0, p):

		 

	ants.append(a)

def classifyAnts(ants):

	clf = LinearDiscriminantAnalysis()


	# Training
	for a in ants:

		X = []
		Y = []

		for i in range(0, num_train):
			train = []

			for n in a.ant_f:
				train.append(train_data[i][n])

			X.append(train)
			Y.append(result_data_train[i])

		X_np = np.array(X)
		Y_np = np.array(Y)
		
		clf.fit(X_np, Y_np)

	# Testing
	for a in ants:

		X = []
		Y = []

		for i in range(0,  num_test):
			test = []
			
			for n in a.ant_f:
				test.append(test_data[i][n])

			X.append(test)
			Y.append(result_data_test[i])

		X_np = np.array(X)
		Y_np = np.array(Y)

		predict = clf.predict(X_np)
		a.mse = mean_squared_error(Y_np, predict)
		print a.mse

def updatePheromoneTrail(ants):

	max_mse = ants[k-1].mse
	common_denominator = max_mse - ants[0].mse

	# Delete the previous list of attributes from which m-p features are chosen
	# and fill it with the union of k best ants feature subset
	del selected_features[:]

	# Calculate
	for i in range(0, k):
		for j in range(0, len(features)):
			if j in ants[i].ant_f:
				features[j].delta = features[j].delta + (max_mse - ants[i].mse)/common_denominator

				#Union of best k ants feature subset
				if j not in selected_features:
					selected_features.append(j)

	# Update
	for f in features:
		f.pheromone = rho*f.pheromone + f.delta
		f.delta = 0



def getKey(a):
	return a.mse

def perform_iteration():

	temp_ants = []

	for i in range(0, num_ants):
		initAnt(temp_ants, ants)

	del ants[:]
	ants = list(temp_ants)
	
	classifyAnts(ants)
	ants = sorted(ants,key=getKey)
	updatePheromoneTrail(ants)
	return ants[0]


##### DECLARATIONS #####

cc = 1
max_iter = 1
k = 4
pp = 8
p = 0
num_ants = 30
num_features = 0
features = []
selected_features = []
data = []
train_data = []
test_data = []
result_data = []
result_data_train = []
result_data_test = []
ants = []
f_ans = 'true'
num_tuples = 0
m = 5
split_ratio = 0.8
rho = 0.75
filename = 'CSV_Version.csv'
######////////////######

getData()

#### SPLIT TRAIN AND TEST DATA ##############
num_train = int(num_tuples*split_ratio)
num_test = num_tuples - num_train
train_data = data[:num_train]
test_data = data[num_train:]
result_data_train = result_data[:num_train]
result_data_test = result_data[num_train:]

######/////////////////////////###########

initFeatures()
mi_fc, mi_ff, cmi_ffc = getMutualInfo(filename)

if num_features <= m:
	print "Number of features less than m...exiting"
	exit()

for i in range(0, num_features):
	selected_features.append(i)

for i in range(0, max_iter):
	selected_ant = perform_iteration()
	print sorted(selected_features)
	# To allow only m-p features, update value of
	# p from 0 to whatever
	if i == 0:
		p = pp