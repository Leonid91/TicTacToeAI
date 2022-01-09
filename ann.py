from pandas import read_csv
from sklearn.model_selection import train_test_split
 
# Chargement du dataset depuis le fichier en paramètre
def load_dataset(filename):
	data = read_csv(filename, header=None)
	dataset = data.values

	data = dataset[:, :-1]
	labels = dataset[:,-1]
	
	# On force tout en string pour que panda puisse les mapper en val numériques
	data = data.astype(str)
	labels = labels.reshape((len(labels), 1))
	return data, labels
 
# load the dataset
data, labels = load_dataset('tic-tac-toe-endgame.csv')
# split into train and test sets
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.33, random_state=1)
# summarize
print('Train', data_train.shape, labels_train.shape)
print('Test', data_test.shape, labels_test.shape)
