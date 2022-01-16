#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
from sklearn.metrics import zero_one_loss
from sklearn import svm
from sklearn.utils import shuffle
from collections import Counter
from imblearn.over_sampling import SMOTE

from scipy.spatial.distance import pdist, cdist, squareform

np.random.seed(42)

def distance(a,b):
	d = np.sqrt(np.square(np.subtract(a[0:2], b[0:2])).sum())     # distanze element wise tra year, title_length
	# print(a[2])
	# print(b[2])
	d += np.bitwise_xor(a[2], b[2]).sum()/len(a[2]) # hamming
	d += np.sqrt(np.square(np.subtract(a[3], b[3])).sum())
	return d

def MinMaxScaling(X_train, X_val, X_test, cols):
	X_train_minmaxscaled = X_train.copy()
	X_val_minmaxscaled = X_val.copy()
	X_test_minmaxscaled = X_test.copy()
	
	for col in cols:
		min = np.min(X_train[col])
		max = np.max(X_train[col])
		range = max - min

		X_train_minmaxscaled[col] = (X_train[col] - min) / range
		X_val_minmaxscaled[col] = (X_val[col] - min) / range
		X_test_minmaxscaled[col] = (X_test[col] - min) / range

	return X_train_minmaxscaled, X_val_minmaxscaled, X_test_minmaxscaled

def balancing(df_train, lower_bound):

	bins_count = df_train.bin_y.value_counts()
	for i in range(len(bins_count)):
		if bins_count[i] <= lower_bound:
			bins_count[i] = lower_bound

	bin_sizes = bins_count.to_dict()

	sm = SMOTE(random_state=43, sampling_strategy=bin_sizes)
	df_train_balanced, _ = sm.fit_resample(df_train, df_train['bin_y'])

	return df_train_balanced

def generateSets(data, n_samples, size):
    samples = []
    if(len(data) >= n_samples * size):
        data = shuffle(data, random_state=43).reset_index(drop=True)
        for s in range(n_samples):
            start = s * size
            end = start + size
            samples.append(data.iloc[start:end])
    else:
        for s in range(n_samples):
            samples.append(data.sample(size, replace=False, ignore_index=True))

    return samples

def RandomSubSets(df, size, n_samples): 
	df_samples = [pd.DataFrame(columns=df.columns) for _ in range(n_samples)]
	for c in df.bin_y.unique():       
		df_class_c = df[df.bin_y == c]
		df_class_sets = generateSets(df_class_c, n_samples, size)
		
		for i in range(n_samples):
			df_samples[i] = df_samples[i].append(df_class_sets[i], ignore_index=True)
	
	return df_samples

def split_XYweights(df):
	y = df['bin_y'].astype('int')
	weights = df['ratings_count']
	X = df.drop(columns=['bin_y', 'rating_mean', 'ratings_count'], axis=1)
	
	return X, y, weights

df_train = pd.read_csv("df_train.csv")
df_test = pd.read_csv("df_test.csv")
df_val = pd.read_csv("df_val.csv")

min_bin_cardinality = df_train.bin_y.value_counts().min()
# df_train_SMOTE = balancing(df_train, min_bin_cardinality)
df_trains = RandomSubSets(df_train, min_bin_cardinality, n_samples=7)

size_C_range = 8
size_gamma_range = 8
nr_configurations = size_gamma_range*size_C_range

C_range = np.logspace(-2, 5, size_C_range)
gamma_range = np.logspace(-5, 2, size_gamma_range)

results = pd.DataFrame(columns=['C', 'gamma', 'loss_ensemble'])
config = 0

for c in C_range:
	for gamma in gamma_range:
		y_val_preds = []
		config += 1
		print(f"************************************** {config} out of {nr_configurations} params' configurations --> C: {c}, gamma: {gamma}")

		print("***********STARTING BAGGING***********")
		for n in range(len(df_trains)):
			print(f"****{n+1}° FIT su {n+1}° sample del train****")

			print("PRE-PROCESSING --> split_XYweights")
			X_train_split, y_train_SVC, weights = split_XYweights(df_trains[n])
			X_val_split, y_val_SVC, _ = split_XYweights(df_val)
			X_test_Ssplit, y_test_SVC, _ = split_XYweights(df_test)

			print("CHANGING DATAFRAME FOR HAMMING")
			X_train_SVC = X_train_split.loc[:,["year", "title_length"]]
			X_train_SVC['genres'] = X_train_split.iloc[:,2:21].values.tolist()
			X_train_SVC['tags'] = X_train_split.iloc[:,22:].values.tolist()

			X_val_SVC = X_val_split.loc[:,["year", "title_length"]]
			X_val_SVC['genres'] = X_val_split.iloc[:,2:21].values.tolist()
			X_val_SVC['tags'] = X_val_split.iloc[:,22:].values.tolist()

			print("PRE-PROCESSING --> MinMaxScaling")
			X_train_SVC, X_val_SVC, X_test_SVC = MinMaxScaling(X_train_SVC, X_val_SVC, X_test_Ssplit, ["year", "title_length"])

			print("FITTING & PREDICTING")
			train_distances = cdist(X_train_SVC.values, X_train_SVC.values, lambda a,b: distance(a,b))
			svc = svm.SVC(kernel="precomputed", C=c, gamma=gamma)
			svc.fit(train_distances, y_train_SVC)

			val_distances = cdist(X_val_SVC.values, X_train_SVC.values, lambda a,b: distance(a,b))
			y_val_pred = svc.predict(val_distances).tolist()
			y_val_preds.append(y_val_pred)

			error = zero_one_loss(y_val_SVC, y_val_pred)
			print(f"LOSS --> {error}")
		print("***********ENDING BAGGING***********")

		# y_val_preds = [[predictions after fit su 1° sample], [predictions after fit su 2° sample], [...], ...]
		# therefore, len(y_val_preds) == len(df_trains)
		# 			 len(y_val_preds[idx]) == len(X_val_SVC) == nr different predictions, with idx the n-th sample
		
		nr_predictions = len(y_val_preds[0])
		y_val_pred_voted = []
		print("VOTING")
		for prediction in range(nr_predictions):
			y_val_pred_voted.append(Counter([item[prediction] for item in y_val_preds]).most_common(1)[0][0])

		loss_ensemble = zero_one_loss(y_val_SVC, y_val_pred_voted)
		print(f"LOSS ENSEMBLE (C: {c}, gamma: {gamma}) --> {loss_ensemble}\n\n")
        
		results = results.append({
			'C': c,
			'gamma': gamma,
			'loss_ensemble': loss_ensemble
		}, ignore_index=True)


results.to_csv("results_undersampling_customKernel.csv", index=False)