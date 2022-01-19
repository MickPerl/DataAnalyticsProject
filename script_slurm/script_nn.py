import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import itertools
import torch
from torch import nn
torch.backends.cudnn.benchmark = False
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support as score

from torchinfo import summary
from textwrap import dedent

import os
import argparse

class MoviesDataset(Dataset):
    def __init__(self):
        df = pd.read_csv("df.csv")
        df = self.cleaning(df)

        X, y, weights = self.split_XYweights(df)

        y = self.discretization(y)

        self.num_classes = y.nunique()
        self.X = torch.FloatTensor(X.values)
        self.y = torch.LongTensor(y)
        self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx], self.weights[idx]

    def split_XYweights(self, df):
        y = df['rating_mean']
        weights = df['ratings_count']
        X = df.drop(columns=['ratings_count', 'rating_mean'], axis=1)
        return X, y, weights

    def cleaning(self, df):
        df.dropna(subset = ['rating_mean'], inplace=True)
        df_without_tags = df[df.iloc[:, 23:-2].isna().all(axis=1)]
        df_without_tags_nor_genres = df_without_tags[df_without_tags['(no genres listed)'] == 1]
        rows_to_be_deleted = df.loc[df["movieId"].isin(df_without_tags_nor_genres["movieId"])].index
        df.drop(rows_to_be_deleted, axis=0, inplace=True)
        df.iloc[:, 23:-2] = df.iloc[:, 23:-2].fillna(0)
        df.drop(['(no genres listed)'], inplace=True, axis=1)
        df_year_without_na = df.year[-pd.isna(df.year)]
        df.year = df.loc[:, 'year'].fillna(np.median(df_year_without_na)).astype('int')
        df.drop('movieId', inplace=True, axis=1)
        df.drop_duplicates(inplace=True)
        return df

    def discretization(self, series):
        return pd.cut(series, bins=5, labels=False)


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, af_first_layer, af_hidden_layers, af_output_layer, num_hidden_layers, dropout, batch_norm):
        super(Feedforward, self).__init__()
    
        model = [nn.Linear(input_size, hidden_size), af_first_layer]

        for i in range(num_hidden_layers):
            model.append(nn.Linear(hidden_size, hidden_size))

            if batch_norm:
                model.append(nn.BatchNorm1d(hidden_size))
            
            model.append(af_hidden_layers)
            
            if dropout != 0:
                model.append(nn.Dropout(dropout))
    

        model.append(nn.Linear(hidden_size, num_classes))

        if af_output_layer :
            model.append(af_output_layer)

        self.model = nn.Sequential(*model)
        

    def forward(self, x):
        return self.model(x)


def train_model(model, criterion, optimizer, data_loader, epochs, n_bad_epochs, device, writer):
	model.train()
	loss_values = []
	n_bad_epochs = n_bad_epochs
	patience = 0
	min_loss = np.Inf
	for epoch in range(epochs):
		losses_current_batch = []
		for batch_idx, samples in enumerate(data_loader):
			data, targets = samples[0].to(device), samples[1].to(device)
			optimizer.zero_grad()

			# Forward pass
			y_pred = model(data)
			# Compute Loss
			if str(criterion) == "CrossEntropyLoss()":
				loss = criterion(y_pred, targets)
			else:	# "KLDivLoss()"
				targets = torch.nn.functional.one_hot(targets, num_classes=5).float()
				loss = criterion(y_pred, targets)

			writer.add_scalar("Loss/train", loss, epoch * len(data_loader) + batch_idx + 1)
			loss_values.append(loss.item())
			losses_current_batch.append(loss.item())

			# Backward pass
			loss.backward()
			optimizer.step()

		loss_current_batch = np.mean(losses_current_batch)

        # If the validation loss is at a minimum
		if loss_current_batch < min_loss:
			# Save the model
			# torch.save(model)
			patience = 0
			min_loss = loss_current_batch
		else:
			patience += 1

		print(f"Epoch: {epoch}\t Mean Loss: {loss_current_batch}\t Current min mean loss: {min_loss}")

		if epoch > 4 and patience > n_bad_epochs:
			print(f"Early stopped at {epoch}-th epoch, since the mean loss over mini-batches didn't decrease during the last {n_bad_epochs} epochs")
			return model, loss_values, epoch

	return model, loss_values, epoch



def test_model(model, data_loader, device, output_dict = False):
	model.eval()
	y_pred = []
	y_test = []
	
	for batch_idx, samples in enumerate(data_loader):
	    data, targets = samples[0].to(device), samples[1].to(device)
	    y_pred.append(model(data))
	    y_test.append(targets)
	y_pred = torch.stack(y_pred).squeeze()
	y_test = torch.stack(y_test).squeeze()
	y_pred = y_pred.argmax(dim=1, keepdim=True).squeeze()
	return classification_report(y_test.cpu(), y_pred.cpu(), zero_division=0, output_dict=output_dict)


def set_reproducibility(seed = 42):
	torch.manual_seed(seed)
	np.random.seed(seed)
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	torch.use_deterministic_algorithms(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))


hyperparams = {
	'num_epochs' : [500],
	'n_bad_epochs': [3],
	'num_hidden_layers' : [1, 3, 5, 7],
	'hidden_size' : [8, 16, 32, 64, 128],
	'batch_size' : [16, 32, 64, 128, 256],
	'af_first_layer' : [nn.Tanh(), nn.LeakyReLU()],
	'af_hidden_layers' : [nn.LeakyReLU()],
	'af_output_layer' : [None, nn.LogSoftmax(dim=1)],
	'loss_function' : [nn.CrossEntropyLoss(), nn.KLDivLoss(reduction = 'batchmean')], 
	'dropout' : [0, 0.2, 0.5],
	'batch_norm' : [False, True],
	'learning_rate' : [0.01, 0.001], 
	'optimizer': ["torch.optim.SGD", "torch.optim.Adam"]	
}


if __name__ == "__main__":
	dataset = MoviesDataset()
	train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=dataset.y, random_state=42)
	train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=dataset.y[train_idx], random_state=42)

	# MinMaxScale training, validation and testing set su year e title_length
	X_train = dataset.X[train_idx]
	X_val = dataset.X[val_idx]
	X_test = dataset.X[test_idx]

	train_year_max = torch.max(X_train[:,1])
	train_year_min = torch.min(X_train[:,1])
	dataset.X[train_idx, 1] = (X_train[:,1] - train_year_min)/(train_year_max - train_year_min)
	dataset.X[val_idx, 1] = (X_val[:,1] - train_year_min)/(train_year_max - train_year_min)
	dataset.X[test_idx, 1] = (X_test[:,1] - train_year_min)/(train_year_max - train_year_min)

	train_title_length_max = torch.max(X_train[:,2])
	train_title_length_min = torch.min(X_train[:,2])
	dataset.X[train_idx, 2] = (X_train[:,2] - train_title_length_min)/(train_title_length_max - train_title_length_min)
	dataset.X[val_idx, 2] = (X_val[:,2] - train_title_length_min)/(train_title_length_max - train_title_length_min)
	dataset.X[test_idx, 2] = (X_test[:,2] - train_title_length_min)/(train_title_length_max - train_title_length_min)

	
	def class_weights(y):
		class_count = torch.bincount(y)
		class_weighting = 1. / class_count
		sample_weights = class_weighting[y]   # sarebbe np.array([weighting[t] for t in y_train])
		return sample_weights

	y_train = dataset.y[train_idx]

	sample_weights = class_weights(y_train)
	sampler_class_frequency = WeightedRandomSampler(sample_weights, len(train_idx))

	# MinMaxScaling ratings_count
	#       weights_train = dataset.weights[train_idx] 
	#       weights_val = dataset.weights[val_idx]
	#       weights_test = dataset.weights[test_idx] 
	#       
	#       weights_train_max = torch.max(weights_train)
	#       weights_train_min = torch.min(weights_train)
	#       dataset.weights[train_idx]  = (weights_train - weights_train_min) / (weights_train_max - weights_train_min)
	#       dataset.weights[val_idx] = (weights_val - weights_train_min) / (weights_train_max - weights_train_min)
	#       dataset.weights[test_idx] = (weights_test - weights_train_min) / (weights_train_max - weights_train_min)
	#       
	#       sampler_ratings_count = WeightedRandomSampler(dataset.weights[train_idx], len(train_idx))

	def dict_configs_from_params_cartesian_product(hyperparams) :
		name_params = list(hyperparams.keys())
		cartesian_product_filtered = []
		cartesian_product_config_params = itertools.product(*hyperparams.values())

		for conf_params in cartesian_product_config_params:
			conf_params_dict = {name_params[i]: conf_params[i] for i in range(len(hyperparams))}
			
			if conf_params_dict['batch_norm'] and conf_params_dict['batch_size'] < 32 : # non ha significatività statistica
				# Skipped config with batch_size < 32 and batch norm, since batches aren't statistically significant.
				continue

			if str(conf_params_dict['loss_function']) == "CrossEntropyLoss()" and conf_params_dict['af_output_layer'] != None:
				# Skipped config with CrossEntropy as loss function and whichever activation function in the output layer,
				# since CrossEntropy always contains SoftMax as activation function of output layer.
				continue

			if str(conf_params_dict['loss_function']) == "KLDivLoss()" and str(conf_params_dict['af_output_layer']) != "LogSoftmax(dim=1)":
				# Skipped config with Kullback-Leibler divergence as loss function and whichever activation function
				# in the output layer other than SoftMax: since Kullback-Leibler divergence works with probability
				# distributions, it's suitable the SoftMax as the activation function of the output layer in that it
				# returns a probability distribution over classes for each feature vector in input.
				continue
			
			if conf_params_dict['dropout'] == 0.5 and conf_params_dict['hidden_size'] < 64 :
				continue

			if conf_params_dict['dropout'] == 0.2 and conf_params_dict['hidden_size'] > 32 :
				continue

			cartesian_product_filtered.append(conf_params_dict)
		
		return cartesian_product_filtered

	def split_configs_params(dict_configs, nr_sets = 4):
		assert len(dict_configs) % nr_sets == 0,  "The number of configs params sets have to be a dividend of the cardinality of all configs."
		print(f"Newly created sets (ratio {nr_sets}:1 to all {len(dict_configs)} configs):")

		for i in range(nr_sets):
			globals()[f"configs_set{i}"] = np.array_split(dict_configs, nr_sets)[i]
			print(f"configs_set{i}")
	

	parser = argparse.ArgumentParser()
	parser.add_argument("--nr_sets",
	                    default=6,
	                    type=int,
	                    help="The number of sets all configurations are splitted in")
	parser.add_argument("--idx_set",
						default=0,
	                    type=int,
	                    help="The index of the desidered set")

	args = parser.parse_args()

	assert args.idx_set < args.nr_sets, f"You can specify a set with an index until {args.nr_sets-1}"

	all_configs = dict_configs_from_params_cartesian_product(hyperparams)
	split_configs_params(all_configs, args.nr_sets)

	config_set = eval(f"configs_set{args.idx_set}")

	set_reproducibility()

	nr_train = len(configs_set0) * args.idx_set
	columns = ["nr_train"] + list(all_configs[0].keys()) + ["epoch_stopped", "loss", "accuracy", "precision", "recall", "f1_score", "support"]
	results = pd.DataFrame(columns=columns)

	for config_params in config_set:	

		nr_train += 1
		print(f"{nr_train}° training with params:")
		pprint(config_params)

		name_run = '__'.join(map(str, config_params))
		writer = SummaryWriter(log_dir=os.path.join('tensorboard_logs', f"{args.idx_set}_out_of_{args.nr_sets - 0}", 'Train_' + str(nr_train), name_run))
		
		train_subset = Subset(dataset, train_idx)
		val_subset=Subset(dataset, val_idx)
		test_subset=Subset(dataset, test_idx)
		train_loader=DataLoader(train_subset, batch_size=config_params['batch_size'], shuffle=False, sampler=sampler_class_frequency, drop_last=True)
		val_loader=DataLoader(val_subset, batch_size=1, shuffle=False, drop_last=True)
		test_loader=DataLoader(test_subset, batch_size=1, shuffle=False, drop_last=True)

		model = Feedforward(
			dataset.X.shape[1],
			config_params['hidden_size'],
			dataset.num_classes,
			config_params['af_first_layer'],
			config_params['af_hidden_layers'],
			config_params['af_output_layer'],
			config_params['num_hidden_layers'],
			config_params['dropout'], 
			config_params['batch_norm'])
		writer.add_graph(model, dataset.X[train_idx])
		model.to(device)
		# summary(model, input_size=(config_params['batch_size'], int(len(config_set) / config_params['batch_size']), 1149), col_names= ["input_size","output_size", "num_params"], verbose=1)
		# dataset.X[train_idx].shape[1] == 1149, dataset.X[train_idx].shape[0] == 35850			provare verbose = 2 per weight e bias
		# test_model(model, val_loader, device)

		loss_func = config_params['loss_function'] 

		optim = eval(config_params['optimizer'] + "(model.parameters(), lr=config_params['learning_rate'])")
		model, loss_values, epoch_stopped = train_model(model, loss_func, optim, train_loader, config_params['num_epochs'], config_params['n_bad_epochs'], device, writer)
		print(f"Loss: {loss_values[-1]}", end="\n\n")
		writer.flush()
		writer.close()

		report = test_model(model, val_loader, device, True)
		index_classes = len(report) - 3
		f1_score = [report[str(i)]['f1-score'] for i in range(index_classes)]
		precision = [report[str(i)]['precision'] for i in range(index_classes)]
		recall = [report[str(i)]['recall'] for i in range(index_classes)]
		support = [report[str(i)]['support'] for i in range(index_classes)]
		accuracy = report['accuracy']
		row_values= [nr_train] + \
			list(config_params.values()) + \
			[epoch_stopped, loss_values[-1], accuracy, precision, recall, f1_score, support]
		results=results.append(pd.Series(row_values, index=columns), ignore_index=True)



	results.to_csv(f"results_nrSets{args.nr_sets}_idxSet{args.idx_set}.csv", index=False)
	