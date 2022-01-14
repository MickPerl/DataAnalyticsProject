#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
from sklearn.metrics import zero_one_loss
from sklearn import svm
from sklearn.utils import shuffle
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def sampling(data, n_samples, size):
	samples = []
	for s in range(n_samples):
		if len(data) >= n_samples * size:
			data = shuffle(data, random_state=43).reset_index(drop=True)
			start = s * size
			end = start + size
			samples.append(data.iloc[start:end])
		else:
			samples.append(data.sample(size, replace=False, ignore_index=True))

	return samples

def MultipleRandomUnderSampling(df, size, n_samples = 10): 
	df_samples = [pd.DataFrame(columns=df.columns) for x in range(n_samples)]
	for c in df.bin_y.unique():       
		df_class_c = df[df.bin_y == c]
		
		df_class_c_samples = sampling(df_class_c, n_samples, size)
		
		for idx_sample in range(n_samples):
			df_samples[idx_sample] = df_samples[idx_sample].append(df_class_c_samples[idx_sample], ignore_index=True)
	
	return df_samples


def split_XYweights(df):
	y = df['bin_y'].astype('int')
	weights = df['ratings_count']
	X = df.drop(columns=['bin_y', 'rating_mean', 'ratings_count'], axis=1)
	
	return X, y, weights

def LDA(X_train, X_val, X_test, y):
	lda = LinearDiscriminantAnalysis()
	lda.fit(X_train, y)

	s = 0
	nr_canonical_variables = 1
	for comp in lda.explained_variance_ratio_:
		s += comp
		if s > 0.95:
			break
		nr_canonical_variables += 1

	X_train = lda.transform(X_train)
	X_train = X_train[:,:nr_canonical_variables]
	X_train = pd.DataFrame(X_train, columns = [f"LD{i}" for i in range(1, X_train.shape[1] + 1)])

	X_val = pd.DataFrame(lda.transform(X_val)[:,:nr_canonical_variables], columns = [f"LD{i}" for i in range(1, X_train.shape[1] + 1)])
	X_test = pd.DataFrame(lda.transform(X_test)[:,:nr_canonical_variables], columns = [f"LD{i}" for i in range(1, X_train.shape[1] + 1)])

	return X_train, X_val, X_test


df_train = pd.read_csv("df_train.csv")
df_test = pd.read_csv("df_test.csv")
df_val = pd.read_csv("df_val.csv")

min_bin_cardinality = df_train.bin_y.value_counts().min()
df_trains = MultipleRandomUnderSampling(df_train, min_bin_cardinality, n_samples=12)

size_C_range = 8
size_gamma_range = 8
nr_configurations = size_gamma_range*size_C_range

C_range = np.logspace(-2, 5, size_C_range)
gamma_range = np.logspace(-5, 2, size_gamma_range)

results = pd.DataFrame(columns=['C', 'gamma', 'error_ensemble'])

config = 0
for c in C_range:
	for gamma in gamma_range:
		config += 1
		y_val_preds = []
		print(f"************************************** {config} out of {nr_configurations} params' configurations --> C: {c}, gamma: {gamma}")

		print("***********STARTING BAGGING***********")
		for n in range(len(df_trains)):
			print(f"****{n+1}° FIT su {n+1}° sample del train****")

			print("PRE-PROCESSING --> split_XYweights")
			X_train_SVC, y_train_SVC, weights = split_XYweights(df_trains[n])
			X_val_SVC, y_val_SVC, _ = split_XYweights(df_val)
			X_test_SVC, y_test_SVC, _ = split_XYweights(df_test)

			print("PRE-PROCESSING --> MinMaxScaling")
			X_train_SVC, X_val_SVC, X_test_SVC = MinMaxScaling(X_train_SVC, X_val_SVC, X_test_SVC, ['title_length','year'])

			print("PRE-PROCESSING --> LDA")
			X_train_SVC, X_val_SVC, X_test_SVC = LDA(X_train_SVC, X_val_SVC, X_test_SVC, y_train_SVC)

			print("FITTING & PREDICTING")
			svc = svm.SVC(kernel="rbf", C=c, gamma=gamma)
			svc.fit(X_train_SVC, y_train_SVC)

			y_val_pred = svc.predict(X_val_SVC).tolist()
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

		error_ensemble = zero_one_loss(y_val_SVC, y_val_pred_voted)
		print(f"LOSS ENSEMBLE (C: {c}, gamma: {gamma}) --> {error_ensemble}\n\n")
        
		results = results.append({
			'C': c,
			'gamma': gamma,
			'error_ensemble': error_ensemble
		}, ignore_index=True)