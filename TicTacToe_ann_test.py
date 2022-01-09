import keras
from keras.models import Sequential, save_model, load_model

# Convertit les val data en val num�rique ordonn�e
def prepare_inputs(data_train, data_test):
	oe = OrdinalEncoder()
	oe.fit(data_train)
	data_train_enc = oe.transform(data_train)
	data_test_enc = oe.transform(data_test)
	return data_train_enc, data_test_enc

# Chargement et compilation du mod�le 
model = load_model('./saved_ann_model', compile = True)

# Test de pr�diction
sample = []
# Consid�rons le board suivant : 
# X | b | O
# X | O | b  
# b | b | b
# O = 1
# X = 2
# b = 0
# Il y a 5 coups possibles pour X
sample.append([2, 2 ,1, 2, 1, 0, 0, 0, 0])
sample.append([2, 0 ,1, 2, 1, 2, 0, 0, 0])
sample.append([2, 0 ,1, 2, 1, 1, 2, 0, 0]) #Devrait �tre la meilleure
sample.append([2, 0 ,1, 2, 1, 1, 0, 2, 0])
sample.append([2, 0 ,1, 2, 1, 1, 0, 0, 2])

# Si le res de la pr�diction est < � 0.5, alors ce n'est pas une bonne solution. Sinon, �a en est une
predictions = (model.predict(sample) > 0.5).astype(int)
print(predictions)
# Output : 0, 0, 1, 0, 0 => Donc l'hypoth�se ci-dessus est valid�e