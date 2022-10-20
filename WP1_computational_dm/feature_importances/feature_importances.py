# -*- coding: utf-8 -*-
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier

def feature_rank(importances, stds, feature_names):
	indices = np.argsort(importances)[::-1]
	print("Feature ranking:")
	#for f in range(len(feature_names)):
		#print("%d. %s: %f (%f)" % (f+1, feature_names[indices[f]], importances[indices[f]], stds[indices[f]]))	
	for f in range(len(feature_names)):
		print("%d." % (f+1))	
	for f in range(len(feature_names)):
		print("%s" % (feature_names[indices[f]]))	
	for f in range(len(feature_names)):
		print("%f" % (importances[indices[f]]))	
	for f in range(len(feature_names)):
		print("(%f)" % (stds[indices[f]]))	
	
def z_score_normalization(X):
	X = (X - np.nanmean(X, axis=0))/np.nanstd(X, axis=0)
	return X

def median_imputation(X):
	for i in range(X.shape[1]):
		X1 = X[:,i]
		X1[np.isnan(X1)] = np.nanmedian(X1)
		X[:,i] = X1
	return X

def kNNImputation(X, k):
	imputer = KNNImputer(missing_values=np.nan, n_neighbors=k)
	return imputer.fit_transform(X)

def get_dictionary(filename):
	file = open(filename, 'rb')
	table = [row.decode('utf8').strip().split(u';') for row in file.readlines()]		
	cnt = 0
	for row in table:
		for i in range(len(row)):
			row[i] = row[i].replace(u',',u'.')
		table[cnt] = row
		cnt = cnt + 1 
	arr = np.array(table)	
	headers = arr[0]
	arr = np.delete(arr, 0, 0)
	arrT = arr.T
	dictionary = {}
	for i in range(len(headers)):
		dictionary[headers[i]] = arrT[i]	
	return dictionary, headers

def main():	
	filename = 'Muutosmatka_changes036.csv'
	dictionary, headers = get_dictionary(filename)
	IDs = [np.nan if item=='#NULL!' or item==' ' else np.double(item) for item in dictionary['ID']]
	paino_suht_muut = [np.nan if item=='#NULL!' or item==' ' else np.double(item) for item in dictionary['Painomuutos036_1104']]
	paino_muut_lk = [np.nan if item=='#NULL!' or item==' ' else np.double(item) for item in dictionary['Painomuutos2LK_36']]
	WBSI_muutos = [np.nan if item=='#NULL!' or item==' ' else np.double(item) for item in dictionary['WBSI_muutos0_36_1104']]
	AAQ_muutos = [np.nan if item=='#NULL!' or item==' ' else np.double(item) for item in dictionary['AAQ_muutos0_36_1104']]
	DASS_muutos = [np.nan if item=='#NULL!' or item==' ' else np.double(item) for item in dictionary['DASS_muutos0_36_1104']]
	GSE_muutos = [np.nan if item=='#NULL!' or item==' ' else np.double(item) for item in dictionary['GSE_muutos0_36_1104']]
	#Yksittäiset muuttujat
	headers = headers[7::]
	X = np.zeros([len(IDs),len(headers)])
	for idx, header in enumerate(headers):
		X[:,idx] = [np.nan if item=='#NULL!' or item==' ' else np.double(item) for item in dictionary[header]]
	X = kNNImputation(X, 10)
	headers1 = headers
	X1 = X
	#WBSI-alakategoriat		
	X_WBSI = X[:,38::] 	
	WBSI_Unwanted = X_WBSI[:,[1,2,3,4,5,6,8,14]]
	WBSI_Unwanted = WBSI_Unwanted.sum(axis=1)
	WBSI_Suppression = X_WBSI[:,[0,7,10,13]]
	WBSI_Suppression = WBSI_Suppression.sum(axis=1)
	WBSI_SelfDist = X_WBSI[:,[9,11,12]]
	WBSI_SelfDist = WBSI_SelfDist.sum(axis=1)	
	#Summamuuttujat
	headers2 = ['WBSI','AAQ','DASS','GSE']
	X2 = np.zeros([len(IDs),len(headers2)])
	X2[:,0] = WBSI_muutos
	X2[:,1] = AAQ_muutos
	X2[:,2] = DASS_muutos
	X2[:,3] = GSE_muutos
	X2 = median_imputation(X2)
	#Lisätään alakategoriat
	headers3 = ['WBSI','AAQ','DASS','GSE','WBSI_Unwanted','WBSI_Suppression','WBSI_SelfDist']
	X3 = np.zeros([len(IDs),len(headers2)+3])
	X3[:,0:4] = X2
	X3[:,4] = WBSI_Unwanted 
	X3[:,5] = WBSI_Suppression
	X3[:,6] = WBSI_SelfDist
	#Z-score-muunnos
	X1 = z_score_normalization(X1)
	X2 = z_score_normalization(X2)
	X3 = z_score_normalization(X3)
	#Painoarvojen laskenta 
	repetitions = 100
	nfolds = 5
	X = X3
	headers = headers3
	y = np.transpose(paino_suht_muut)		
	#y = np.transpose(paino_muut_lk)
	#Verkkohaku
	#parameters = {'max_features':[3, 5, 6, 7, 8, 11], 'min_samples_leaf':[1, 5, 8]}
	#forest = RandomForestRegressor(random_state=0)
	#clf = GridSearchCV(forest, parameters, cv=5)
	#clf.fit(X1, y)
	#print(clf.best_params_)		
	results = np.zeros((repetitions, len(headers)))
	for i in range(repetitions):
		print ("Repetition: %d" % (i))
		kf = KFold(n_splits=nfolds, random_state=i, shuffle=True)
		for train_index, test_index in kf.split(X):
			#forest = RandomForestRegressor(random_state=i)
			#forest = RandomForestClassifier(random_state=i)
			forest = ExtraTreesRegressor(random_state=i)
			#forest = ExtraTreesClassifier(random_state=i)
			X_train = X[train_index,:]
			y_train = y[train_index]
			forest.fit(X_train, y_train)
			importances = forest.feature_importances_
			results[i][:]  = results[i][:] + importances
		results[i][:] = results[i][:] / (1.0*nfolds)	
	stds = np.std(results, axis=0)
	results = np.mean(results, axis=0)
	feature_rank(results, stds, headers)	
	
	
if __name__ == "__main__":
	main()


