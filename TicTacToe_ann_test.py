import keras
from keras.models import Sequential, save_model, load_model

# Chargement et compilation du modèle 
model = load_model('./saved_ann_model', compile = True)

# Test de prédiction 1
sample = []
# Considérons le board suivant : 
# X |   | O
# X | O |   
#   |   | 
# O = 1
# X = 2
#   = 0
# Il y a 5 coups possibles pour X
sample.append([2, 2 ,1, 2, 1, 0, 0, 0, 0])
sample.append([2, 0 ,1, 2, 1, 2, 0, 0, 0])
sample.append([2, 0 ,1, 2, 1, 1, 2, 0, 0]) #Devrait être la meilleure
sample.append([2, 0 ,1, 2, 1, 1, 0, 2, 0])
sample.append([2, 0 ,1, 2, 1, 1, 0, 0, 2])

# Si le res de la prédiction est < à 0.5, alors ce n'est pas une bonne solution. Sinon, ça en est une
#predictions = (model.predict(sample) > 0.5).astype(int)
predictions = model.predict(sample)
print(predictions)
# Output : 0, 0, 1, 0, 0 => Donc l'hypothèse ci-dessus est validée

# Test de prédiction 2
sample = []
# Considérons le board suivant : 
# X | X | O
#   | O |    
#   |   | 

# Il y a 5 coups possibles pour X
sample.append([2, 2 ,1, 2, 1, 0, 0, 0, 0])
sample.append([2, 2 ,1, 0, 1, 2, 0, 0, 0])
sample.append([2, 2 ,1, 0, 1, 0, 2, 0, 0]) #Devrait être la meilleure
sample.append([2, 2 ,1, 0, 1, 0, 0, 2, 0])
sample.append([2, 2 ,1, 0, 1, 0, 0, 0, 2])

# Si le res de la prédiction est < à 0.5, alors ce n'est pas une bonne solution. Sinon, ça en est une
#predictions = (model.predict(sample) > 0.5).astype(int)
predictions = model.predict(sample)
print(predictions)
# Output : 0, 0, 1, 0, 0 => Donc l'hypothèse ci-dessus est validée