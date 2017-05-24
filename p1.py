import csv
from random import randint
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, accuracy_score
from get_mi import get_mutual_information
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sets import Set
import math
import sys


class feature(object):

	def __init__(self):
		self.delta = 0
		self.pheromone = 0


class ant(object):

	def __init__(self):
		self.ant_accuracy = 0.0
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

	x = sys.float_info.max
	global num_features
	global gamma
	global beta

	for i in a.ant_f:
		term1 = (mi_ff[f][f] - mi_ff[f][i])/mi_ff[f][f]
		x = min(x, term1)

	term2 = beta/len(a.ant_f)
	sum = 0.0
	for i in a.ant_f:
		sum = sum + math.pow(cmi_ffc[f][i]/(mi_fc[f] + mi_fc[i]), gamma)
	
	term2 = term2*sum
	return term1*term2

def local_importance(a, f):

	Di = calculate_Di(a, f)
	Li = mi_fc[f]*(2/(1+math.exp(-1*alpha*Di)) - 1)
	return Li


def calculate_denominator(a):

	den = 0.0
	for ff in range(0, len(features)):
		if ff not in a.ant_f:
			den = den + features[ff].pheromone*local_importance(a, ff)
	
	return den


def calculate_usm(a, f, den):

	usm = features[f].pheromone*local_importance(a, f)/den
	return usm

def initAnt(temp_ants):

	a = ant()
	temp_selected_features = list(selected_features)

	for i in range(0, m-p):

		r = randint(0, len(temp_selected_features)-1)
		a.ant_f.append(temp_selected_features[r])
		temp_selected_features.remove(temp_selected_features[r])

		
	for i in range(0, p):
		
		den = calculate_denominator(a)
		max_usm = sys.float_info.min
		max_usm_feature_number = -1
		for f in range(0, len(features)):
			
			if f not in a.ant_f:
				usmmmm = calculate_usm(a, f, den)
				if max_usm < usmmmm:
					max_usm = usmmmm
					max_usm_feature_number = f
		a.ant_f.append(max_usm_feature_number)

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
		a.ant_accuracy = accuracy_score(Y_np, predict, normalize=False)*100.0/len(Y_np)

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
	
	
	# Remove extra features
	unique_k_set = Set([])
	for i in range(0, k):
		for f in ants[i].ant_f:
			unique_k_set.add(f)
	
	del selected_features[:]
	for f in unique_k_set:
		selected_features.append(f)
	print "size of selected features is ", len(selected_features)

def getKey(a):
	return a.mse
def getAccuracyKey(a):
	return a.ant_accuracy

def perform_iteration():
	
	temp_ants = []
	for i in range(0, num_ants):
		initAnt(temp_ants)
	
	print "initialized ants"
	
	#del ants[:]
	ants = list(temp_ants)

	classifyAnts(ants)
	acc_ant = sorted(ants, key = getAccuracyKey, reverse = True)
	print acc_ant[0].ant_accuracy
	ants = sorted(ants,key=getKey)
	updatePheromoneTrail(ants)
	return ants[0]

##### DECLARATIONS #####

cc = 1
max_iter = 15
k = 10
pp = 3
p = 0
num_ants = 10
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
m = 8
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

train_data, test_data, result_data_train, result_data_test = train_test_split(data, result_data, test_size = 0.2, random_state = 42)
num_train = len(train_data)
num_test = len(test_data)
print "number train and test", num_train, num_test 
######/////////////////////////###########

initFeatures()
print "initialized features"
#mi_fc, mi_ff, cmi_ffc = get_mutual_information('jm1_final.csv')
with open('mi_fc.pkl', 'rb') as pick:
	mi_fc = pickle.load(pick)
pick.close()
with open('mi_ff.pkl', 'rb') as pick:
	mi_ff = pickle.load(pick)
pick.close()
# with open('cmi_ffc.pkl', 'rb') as pick:
# 	cmi_ffc = pickle.load(pick)
# pick.close()
cmi_ffc = get_mutual_information('jm1_final.csv')
print "fetched mutual info"
if num_features <= m:
	print "Number of features less than m"
	exit()

for i in range(0, num_features):
	selected_features.append(i)

print "features are"
print selected_features

for i in range(0, max_iter):
	perform_iteration()
	if i == 0:
		p = pp