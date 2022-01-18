
# TODO: da fare ancora
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
        series = pd.cut(series, bins=5, labels=False)
        return series



class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, af_first_layer, af_hidden_layers, af_output_layer, num_hidden_layers, dropout):
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



def train_model(model, criterion, optimizer, epochs, data_loader, device):
    model.train()
    loss_values = []
    n_epochs_stop = 3
    patience = 0
    early_stop = False
    min_loss = np.Inf
    for epoch in range(epochs):
        losses_current_batch = []
        for batch_idx, samples in enumerate(data_loader):
            data, targets = samples[0].to(device), samples[1].to(device)
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(data)
            # y_predd = torch.argmax(y_pred, dim=1)

            # Compute Loss
            loss = criterion(y_pred, targets)
            writer.add_scalar("Loss/train", loss, epoch * len(data_loader) + batch_idx +1)
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
        
        if epoch > 4 and patience > n_epochs_stop:
            early_stop = True
            print(f"Early stopped at {epoch}-th epoch, since the mean loss over mini-batches didn't decrease during the last {n_epochs_stop} epochs")
            break

    return model, loss_values, epoch



def evaluate_model(model, data_loader, device):
	model.eval()
	y_pred = []
	y_val = []	
	for batch_idx, samples in enumerate(data_loader):
	    data, targets = samples[0].to(device), samples[1].to(device)
	    y_pred.append(model(data))
	    y_val.append(targets)
	y_pred = torch.stack(y_pred).squeeze()
	y_val = torch.stack(y_val).squeeze()
	y_pred = y_pred.argmax(dim=1, keepdim=True).squeeze()
	# classification_report(y_val.cpu(), y_pred.cpu(), zero_division=0)
	report = classification_report(y_val.cpu(), y_pred.cpu(), zero_division=0, output_dict=True)
	return report


def test_model(model, data_loader, device):
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
    print(classification_report(y_test.cpu(), y_pred.cpu(), zero_division=0))


def set_reproducibility(seed = 42):
	torch.manual_seed(seed)
	np.random.seed(seed)
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	torch.use_deterministic_algorithms(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))


hyperparams = {
	'num_epochs' : [500],
	'num_hidden_layers' : [1, 3, 5, 7, 10],
	'hidden_size' : [8, 16, 32, 64, 128],
	'batch_size' : [16, 32, 64, 128, 256, 512],
	'af_first_layer' : [nn.Tanh(), nn.LeakyReLU()],
	'af_hidden_layers' : [nn.LeakyReLU()],
	'af_output_layer' : [None], # [None, nn.Softmax(dim=1)],
	'loss_function' : [nn.CrossEntropyLoss()], #[nn.CrossEntropyLoss(), nn.KLDivLoss(reduction = 'batchmean')],
	'dropout' : [0, 0.2, 0.4],
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

	
	# Creating samplers to manage unbalancing classes
	def class_weights(y):
		class_count = torch.bincount(y)
		class_weighting = 1. / class_count
		sample_weights = class_weighting[y]   # sarebbe np.array([weight[t] for t in y_train])
		return sample_weights

	y_train = dataset.y[train_idx]

	sample_weights = class_weights(y_train)
	sampler_class_frequency = WeightedRandomSampler(sample_weights, len(train_idx))

	# MinMaxScaling ratings_count
	weights_train = dataset.weights[train_idx] 
	weights_val = dataset.weights[val_idx]
	weights_test = dataset.weights[test_idx] 

	weights_train_max = torch.max(weights_train)
	weights_train_min = torch.min(weights_train)
	dataset.weights[train_idx]  = (weights_train - weights_train_min) / (weights_train_max - weights_train_min)
	dataset.weights[val_idx] = (weights_val - weights_train_min) / (weights_train_max - weights_train_min)
	dataset.weights[test_idx] = (weights_test - weights_train_min) / (weights_train_max - weights_train_min)

	sampler_ratings_count = WeightedRandomSampler(dataset.weights[train_idx], len(train_idx))

	set_reproducibility()

	nr_train = 0
	*names, = hyperparams
	columns = ["nr_train"] + names + ["epoch_stopped", "loss", "accuracy", "precision", "recall", "f1_score", "support"]
	results = pd.DataFrame(columns=columns)

	for config_params in itertools.product(*hyperparams.values()):
		for param in range(len(hyperparams)):
			globals()[names[param]] = config_params[param]

		if batch_norm and batch_size < 32 : # non ha significatività statistica
			continue

		if loss_function == nn.CrossEntropyLoss() and af_output_layer != None:	# quando c'è crossentropy, nessuna funzione di attivazione
			continue

		nr_train += 1
		print(f"{nr_train}° training with params:")
		for param in range(len(hyperparams)):
			print(f"-- {names[param]}: {config_params[param]}")

		writer = SummaryWriter(log_dir=os.path.join("tensorboard_logs", "Train n. " + str(nr_train) + " --> " + " - ".join(map(str, config_params))))
		
		train_subset = Subset(dataset, train_idx)
		val_subset=Subset(dataset, val_idx)
		test_subset=Subset(dataset, test_idx)
		train_loader=DataLoader(train_subset, batch_size=batch_size, shuffle=False, sampler=sampler_class_frequency, drop_last=True, num_workers = 1, pin_memory = True)
		val_loader=DataLoader(val_subset, batch_size=1, shuffle=False, drop_last=True, num_workers = 1, pin_memory = True)
		test_loader=DataLoader(test_subset, batch_size=1, shuffle=False, drop_last=True, num_workers = 1, pin_memory = True)

		model=Feedforward(dataset.X.shape[1], hidden_size, dataset.num_classes, af_first_layer, af_hidden_layers, af_output_layer, num_hidden_layers, dropout)
		writer.add_graph(model, dataset.X)
		model.to(device)
		# evaluate_model(model, val_loader, device)


		model, loss_values, epoch_stopped = train_model(model, loss_function, eval(optimizer + "(model.parameters(), lr=learning_rate)"), num_epochs, train_loader, device)
		print(f"Loss: {loss_values[-1]}", end="\n\n")
		writer.flush()
		writer.close()

		report = evaluate_model(model, val_loader, device)
		index_classes = len(report) - 3
		f1_score = [report[str(i)]['f1-score'] for i in range(index_classes)]
		precision = [report[str(i)]['precision'] for i in range(index_classes)]
		recall = [report[str(i)]['recall'] for i in range(index_classes)]
		support = [report[str(i)]['support'] for i in range(index_classes)]
		accuracy = report['accuracy']
		row_values= [nr_train] + list(map(str, config_params)) + [epoch_stopped, loss_values[-1], accuracy, precision, recall, f1_score, support]
		results=results.append(pd.Series(row_values, index=columns), ignore_index=True)
		# plt.plot(loss_values)
		# plt.title("Number of epochs: {}".format(num_epochs))
		# plt.show()

	

	results.to_csv("results.csv", index=False)


