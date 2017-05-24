import csv
from random import randint
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
from get_mi import get_mutual_information
import pandas as pd


class feature(object):

	def __init__(self):
		self.delta = 0
		self.pheromone = 0


class ant(object):

	def __init__(self):
		self.ant_f = []
		self.mse = 0

def getData():

	temp_list = []
	global num_tuples
	global num_features
	global filename
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
	print num_features, num_tuples


def initFeatures():
	for i in range(0, num_features):
		temp = feature()
		temp.delta = 0
		temp.pheromone = cc
		features.append(temp)


def calculate_Di(a, f):

	x = 9999.9
	global num_features
	global gamma
	global beta

	for i in range(0, num_features):
		if i in a.ant_f:
			term1 = (mi_ff[f][f] - mi_ff[f][i])/mi_ff[f][f]
			x = min(x, term1)

	term2 = beta/len(a.ant_f)
	sum = 0.0
	for i in a.ant_f:
		sum += math.pow(cmi_ffc[f][i]/(mi_fc[f] - mi_fc[i]), gamma)
	
	term2 = term2*sum
	return term1*term2

def local_importance(a, f):

	Di = calculate_Di(a, f)
	Li = mi_fc[f]*(2/(1+math.exp(-1*alpha*Di)) - 1)
	return Li


def calculate_denominator(a, f):

	den = 0.0
	for f in features:
		if f not in a.ant_f:
			den = den + features[f].pheromone*local_importance(a, f)
	
	return den

def initAnt(temp_ants):

	a = ant()
	temp_selected_features = list(selected_features)

	for i in range(0, m):

		r = randint(0, len(temp_selected_features)-1)
		a.ant_f.append(temp_selected_features[r])
		temp_selected_features.remove(temp_selected_features[r])
	
	for a in temp_ants:
		for i in range(0, p):
			den = calculate_denominator(a, f)
			for f in range(0, len(features)):
				if f not in a.ant_f:
					calculate_usm(a, f, den)

	
	temp_ants.append(a)

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

	# Calculate
	for i in range(0, k):
		for j in range(0, len(features)):
			if j in ants[i].ant_f:
				features[j].delta = features[j].delta + (max_mse - ants[i].mse)/common_denominator

	# Update
	for f in features:
		f.pheromone = rho*f.pheromone + f.delta
		f.delta = 0



def getKey(a):
	return a.mse

def perform_iteration():
	
	temp_ants = []
	for i in range(0, num_ants):
		initAnt(temp_ants)
	
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
pp = 2
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
m = 12
split_ratio = 0.8
rho = 0.75
filename = 'CSV_Version.csv'
mu = 1
kappa = 1
alpha = 0.3
gamma = 3
beta = 1.65
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
mi_fc, mi_ff, cmi_ffc = get_mutual_information('jm1_final.csv')

if num_features <= m:
	print "Number of features less than m"
	exit()

for i in range(0, num_features):
	selected_features.append(i)

for i in range(0, max_iter):
	perform_iteration()