import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
 
# Chargement du dataset depuis le fichier en param�tre
def load_dataset(filename):
	data = read_csv(filename, header=None)
	dataset = data.values

	data = dataset[:, :-1]
	labels = dataset[:,-1]
	
	# On force tout en string pour que panda puisse les mapper en val num�riques
	data = data.astype(str)
	labels = labels.reshape((len(labels), 1))
	return data, labels

# Convertit les vals cible en class binaire
def prepare_targets(labels_train, labels_test):
	le = LabelEncoder()
	le.fit(labels_train)
	labels_train_enc = le.transform(labels_train)
	labels_test_enc = le.transform(labels_test)
	return labels_train_enc, labels_test_enc

# Convertit les val data en val num�rique ordonn�e
def prepare_inputs(data_train, data_test):
	oe = OrdinalEncoder()
	oe.fit(data_train)
	data_train_enc = oe.transform(data_train)
	data_test_enc = oe.transform(data_test)
	return data_train_enc, data_test_enc

# Charge le dataset
data, labels = load_dataset('tic-tac-toe-endgame.csv')
# Split le dataset
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.33, random_state=1)
#DEBUG
print('Train size', data_train.shape, labels_train.shape)
print('Test size', data_test.shape, labels_test.shape)

# Prepare les data en les convertissant les labels en int
data_train_enc, data_test_enc = prepare_inputs(data_train, data_test)
labels_train_enc, labels_test_enc = prepare_targets(labels_train, labels_test)

# Construction du r�seau
model = Sequential()
# Hidden layer : 3 neuronnes et on sp�cifie que l'input layer est compos�e de 9 variables
model.add(Dense(200, input_dim=9, activation='relu'))
# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compilation du mod�le
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrainement du mod�le
model.fit(data_train_enc, labels_train_enc, epochs=100, batch_size=32, verbose=2)

# Evaluation de la pr�cision du mod�le
_, accuracy = model.evaluate(data_test_enc, labels_test_enc, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))
