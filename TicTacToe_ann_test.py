import keras
from keras.models import Sequential, save_model, load_model

# Chargement et compilation du mod�le 
model = load_model('./saved_ann_model', compile = True)

# Test de pr�diction 1
sample = []
# Consid�rons le board suivant : 
# X |   | O
# X | O |   
#   |   | 
# O = 1
# X = 2
#   = 0
# Il y a 5 coups possibles pour X
sample.append([2, 2 ,1, 2, 1, 0, 0, 0, 0])
sample.append([2, 0 ,1, 2, 1, 2, 0, 0, 0])
sample.append([2, 0 ,1, 2, 1, 1, 2, 0, 0]) #Devrait �tre la meilleure
sample.append([2, 0 ,1, 2, 1, 1, 0, 2, 0])
sample.append([2, 0 ,1, 2, 1, 1, 0, 0, 2])

# Si le res de la pr�diction est < � 0.5, alors ce n'est pas une bonne solution. Sinon, �a en est une
#predictions = (model.predict(sample) > 0.5).astype(int)
predictions = model.predict(sample)
print(predictions)
# Output : 0, 0, 1, 0, 0 => Donc l'hypoth�se ci-dessus est valid�e

# Test de pr�diction 2
sample = []
# Consid�rons le board suivant : 
# X | X | O
#   | O |    
#   |   | 

# Il y a 5 coups possibles pour X
sample.append([2, 2 ,1, 2, 1, 0, 0, 0, 0])
sample.append([2, 2 ,1, 0, 1, 2, 0, 0, 0])
sample.append([2, 2 ,1, 0, 1, 0, 2, 0, 0]) #Devrait �tre la meilleure
sample.append([2, 2 ,1, 0, 1, 0, 0, 2, 0])
sample.append([2, 2 ,1, 0, 1, 0, 0, 0, 2])

# Si le res de la pr�diction est < � 0.5, alors ce n'est pas une bonne solution. Sinon, �a en est une
#predictions = (model.predict(sample) > 0.5).astype(int)
predictions = model.predict(sample)
print(predictions)
# Output : 0, 0, 1, 0, 0 => Donc l'hypoth�se ci-dessus est valid�e