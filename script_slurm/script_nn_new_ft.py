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
from torch.utils.tensorboard.summary import hparams
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support as score

from torchinfo import summary
from textwrap import dedent

import os

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


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def train_model(model, criterion, optimizer, data_loader, epochs, n_bad_epochs, device, tb, cardinality_training_set):
	model.train()

	loss_values = []	# to store loss values over all batches regardless distinct epochs: it's the list we return after training
	# loss_values_every_epoch = []
	# accuracy_every_epoch = []

	n_bad_epochs = n_bad_epochs
	patience = 0
	min_loss = np.Inf


	for epoch in range(epochs):
		losses_batches_current_epoch = []	# to store loss values over all batches with regard to a single epoch to checking condition about early stopping
		correct_batches_current_epoch = []
		
		for batch_idx, samples in enumerate(data_loader):
			data, targets = samples[0].to(device), samples[1].to(device)
			optimizer.zero_grad()

			# Forward pass
			
			y_pred = model(data)
			# Compute Loss
			if str(criterion) == "CrossEntropyLoss()":
				loss = criterion(y_pred, targets)
			else:	# "KLDivLoss()"
				targets_one_hot_encoded = torch.nn.functional.one_hot(targets, num_classes=5).float()
				loss = criterion(y_pred, targets_one_hot_encoded)

			correct = get_num_correct(y_pred, targets)
			
			tb.add_scalar("Loss every batch", loss, epoch * len(data_loader) + batch_idx + 1)
			tb.add_scalar("Correct every batch", correct, epoch * len(data_loader) + batch_idx + 1)
			tb.add_scalar("Accuracy every batch", correct / len(data), epoch * len(data_loader) + batch_idx + 1)

			loss_values.append(loss.item())
			losses_batches_current_epoch.append(loss.item())
			correct_batches_current_epoch.append(correct)

			# Backward pass
			loss.backward()

			# Looking for batch containing bad samples which cause inf/nan gradients or loss:
			# ideally, this samples should to be removed, but in this case we want only automatically
			# skip them and continue training.
			valid_gradients = True
			for name, param in model.named_parameters():
				if param.grad is not None:
					if torch.isnan(param.grad).any():
						print(f"{name} is nan, so model parameters are not going to be updated: this batch is skipped and the gradient is reset.")
						optimizer.zero_grad()
						valid_gradients = False
					if torch.isinf(param.grad).any():
						print(f"{name} is inf, so model parameters are not going to be updated: this batch is skipped and the gradient is reset.")
						optimizer.zero_grad()
						valid_gradients = False
			if not valid_gradients :
				break

			optimizer.step()
			
			# for tag, value in model.named_parameters():
			# 	tag = tag.replace('.', '/')
			# 	tb.add_histogram('every batch_' + tag, value.data.cpu().detach().numpy(), batch_idx + 1)
			# 	tb.add_histogram('every batch_' + tag + '/grad', value.grad.data.cpu().numpy(), batch_idx + 1)


		total_loss_current_epoch = np.sum(losses_batches_current_epoch)
		tb.add_scalar("Loss every epoch", total_loss_current_epoch, epoch)

		# loss_values_every_epoch.append(total_loss_current_epoch)
		
		total_correct_current_epoch = np.sum(correct_batches_current_epoch)
		tb.add_scalar("Correct every epoch", total_correct_current_epoch, epoch)

		accuracy_current_epoch = total_correct_current_epoch / cardinality_training_set
		tb.add_scalar("Accuracy every epoch", accuracy_current_epoch, epoch)

		# accuracy_every_epoch.append(accuracy_current_epoch)

		# for tag, value in model.named_parameters():
		# 	tag = tag.replace('.', '/')
		# 	tb.add_histogram('every epoch_' + tag, value.data.cpu().detach().numpy(), epoch)
		# 	tb.add_histogram('every epoch_' + tag + '/grad', value.grad.data.cpu().numpy(), epoch)

		mean_loss_current_epoch = np.mean(losses_batches_current_epoch)

		if epoch < 5 :
			print(f"Epoch: {epoch}\t Total loss: {total_loss_current_epoch}")
			continue
		
		if epoch == 5 :
			print("Waiting for three consecutive epochs during which the mean loss over batches does not decrease...")
        
		if mean_loss_current_epoch < min_loss:
			# Save the model
			# torch.save(model)
			patience = 0
			min_loss = mean_loss_current_epoch
		else:
			patience += 1

		print(f"Epoch: {epoch}\t Total loss: {total_loss_current_epoch}\t Mean Loss: {mean_loss_current_epoch}\t Current min mean loss: {min_loss}")

		if patience == n_bad_epochs:
			print(f"Early stopped at {epoch}-th epoch, since the mean loss over batches didn't decrease during the last {n_bad_epochs} epochs")
			return model, loss_values, epoch, total_loss_current_epoch, accuracy_current_epoch # loss_values_every_epoch, accuracy_every_epoch
			# At the return moment,
			# 		total_loss_current_epoch is the loss value of the last epoch

	return model, loss_values, epoch, total_loss_current_epoch, accuracy_current_epoch 



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

class SummaryWriter(SummaryWriter):

	def add_hparams(self, hparam_dict, metric_dict):
		torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
		if type(hparam_dict) is not dict or type(metric_dict) is not dict:
			raise TypeError('hparam_dict and metric_dict should be dictionary.')
		exp, ssi, sei = hparams(hparam_dict, metric_dict)

		self.file_writer.add_summary(exp)
		self.file_writer.add_summary(ssi)
		self.file_writer.add_summary(sei)
		for k, v in metric_dict.items():
			if v is not None:
				self.add_scalar(k, v)

def class_weights(y):
	class_count = torch.bincount(y)
	class_weighting = 1. / class_count
	sample_weights = class_weighting[y]   # sarebbe np.array([weighting[t] for t in y_train])
	return sample_weights

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

		#######################
		if conf_params_dict['num_hidden_layers'] != 10 or conf_params_dict['batch_size'] != 512 or conf_params_dict['learning_rate'] != 1e-5:
			continue

		cartesian_product_filtered.append(conf_params_dict)
	
	return cartesian_product_filtered

def split_configs_params(dict_configs, nr_sets = 4):
	assert len(dict_configs) % nr_sets == 0,  "The number of configs params sets have to be a dividend of the cardinality of all configs."
	print(f"Newly created sets (ratio {nr_sets}:1 to all {len(dict_configs)} configs):")

	for i in range(nr_sets):
		globals()[f"configs_set{i}"] = np.array_split(dict_configs, nr_sets)[i]
		print(f"configs_set{i}")

if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("Device: {}".format(device))


	new_hyperparams = {
		'num_epochs' : [500],
		'n_bad_epochs': [3],
		'num_hidden_layers' : [3, 5, 7, 10],
		'hidden_size' : [16, 64, 128, 256],
		'batch_size' : [16, 64, 256, 512],
		'af_first_layer' : [nn.Tanh(), nn.LeakyReLU()],
		'af_hidden_layers' : [nn.LeakyReLU()],
		'af_output_layer' : [None, nn.LogSoftmax(dim=1)],
		'loss_function' : [nn.CrossEntropyLoss(), nn.KLDivLoss(reduction = 'batchmean')], 
		'dropout' : [0, 0.5],
		'batch_norm' : [False, True],
		'learning_rate' : [0.01, 1e-5], 
		'optimizer': ["torch.optim.SGD", "torch.optim.Adam"],
		'weight_decay': [1e-4]		
	}


	dataset = MoviesDataset()
	train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=dataset.y, random_state=42)
	train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=dataset.y[train_idx], random_state=42)

	# MinMaxScale training, validation and testing set su year e title_length
	X_train = dataset.X[train_idx]
	X_val = dataset.X[val_idx]
	X_test = dataset.X[test_idx]

	train_year_max = torch.max(X_train[:,0])
	train_year_min = torch.min(X_train[:,0])
	dataset.X[train_idx, 0] = (X_train[:,0] - train_year_min)/(train_year_max - train_year_min)
	dataset.X[val_idx, 0] = (X_val[:,0] - train_year_min)/(train_year_max - train_year_min)
	dataset.X[test_idx, 0] = (X_test[:,0] - train_year_min)/(train_year_max - train_year_min)

	train_title_length_max = torch.max(X_train[:,1])
	train_title_length_min = torch.min(X_train[:,1])
	dataset.X[train_idx, 1] = (X_train[:,1] - train_title_length_min)/(train_title_length_max - train_title_length_min)
	dataset.X[val_idx, 1] = (X_val[:,1] - train_title_length_min)/(train_title_length_max - train_title_length_min)
	dataset.X[test_idx, 1] = (X_test[:,1] - train_title_length_min)/(train_title_length_max - train_title_length_min)

	y_train = dataset.y[train_idx]

	sample_weights = class_weights(y_train)
	sampler_class_frequency = WeightedRandomSampler(sample_weights, len(train_idx))


	all_configs = dict_configs_from_params_cartesian_product(new_hyperparams)

	
	nr_train = 5761

	set_reproducibility()	
		
	columns = ["nr_train"] + list(all_configs[0].keys()) + ["epoch_stopped", "loss", "accuracy", "precision", "precision_total", "recall", "recall_total", "f1_score", "f1_score_total", "support"]
	results = pd.DataFrame(columns=columns)

	for config_params in all_configs:
		nr_train += 1
		print(f"{nr_train}° training with params:")
		pprint(config_params)

		list_params_config = list(map(str, list(config_params.values())))
		name_run = '__'.join(list_params_config)
		with SummaryWriter(log_dir=os.path.join('tensorboard_logs', f"new_fine_tuning", 'Train_' + str(nr_train), name_run)) as tb:
		# tb = SummaryWriter(log_dir=os.path.join('tensorboard_logs', f"{idx_set}_out_of_{nr_sets - 1}", 'Train_' + str(nr_train), name_run))

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

			model.to(device)
			# input_model = dataset.X[train_idx][:config_params['batch_size']].to(device)
			# tb.add_graph(model, input_model)

			# summary(model, input_size=(config_params['batch_size'], int(35850 // config_params['batch_size']), 1149), col_names= ["input_size","output_size", "num_params"], verbose=1)
			# dataset.X[train_idx].shape[1] == 1149, dataset.X[train_idx].shape[0] == 35850			provare verbose = 2 per weight e bias
			# test_model(model, val_loader, device)

			loss_func = config_params['loss_function'] 

			optim = eval(config_params['optimizer'] + "(model.parameters(), lr=config_params['learning_rate'])")

			cardinality_training_set = len(X_train)
			model, loss_values, epoch_stopped, loss_value_last_epoch, accuracy_last_epoch = train_model(model, loss_func, optim, train_loader, config_params['num_epochs'], config_params['n_bad_epochs'], device, tb, cardinality_training_set)
			
			print(f"Loss: {loss_value_last_epoch}", end="\n\n")

			report = test_model(model, val_loader, device, True)
			index_classes = len(report) - 3
			f1_score = [float(report[str(i)]['f1-score']) for i in range(index_classes)]
			f1_score_total = np.sum(f1_score)
			precision = [float(report[str(i)]['precision']) for i in range(index_classes)]
			precision_total = np.sum(precision)
			recall = [float(report[str(i)]['recall']) for i in range(index_classes)]
			recall_total = np.sum(recall)
			support = [int(report[str(i)]['support']) for i in range(index_classes)]
			accuracy = report['accuracy']


			row_values= [nr_train] + list_params_config + [epoch_stopped, loss_value_last_epoch, accuracy, precision, precision_total, recall, recall_total, f1_score, f1_score_total, support]
			results=results.append(pd.Series(row_values, index=columns), ignore_index=True)
			# plt.plot(loss_values)
			# plt.title("Number of epochs: {}".format(num_epochs))
			# plt.show()

			dict_params_config = {list(config_params.keys())[z]: list_params_config[z] for z in range(len(config_params))}
			tb.add_hparams(hparam_dict = dict_params_config, metric_dict = {"Accuracy every epoch": None, "Loss every epoch": None})
			tb.flush()
			tb.close()
		del model, optim, train_loader, val_loader
	

	
	results.to_csv("results_NN.csv", index=False)
	
	torch.cuda.empty_cache()
	