from keras.datasets import cifar100                 #pour importer la base cifar100
import pandas                                       #bibliothèque utilisée pour la manipulation de données 
from sklearn.preprocessing import LabelBinarizer    #pour utiliser la fonction LabelBinarizer (ligne 23)
from keras.models import Sequential                 #ligne 31
from keras.layers.core import Dense                 #ligne 33


##Chargement des données à partir de la bibliothèque Keras
#Chargement des données avec les superclasses (coarse)
print("Chargement des données CIFAR-100...")
split_c = cifar100.load_data(label_mode='coarse')
((trainX_c, trainY_c), (testX_c, testY_c)) = split_c
trainX_c = trainX_c.astype("float") / 255.0         #échantillon d'apprentissage
testX_c = testX_c.astype("float") / 255.0           #échantillon test

#Chargement des données avec les classes (fine)
split_f = cifar100.load_data(label_mode='fine')
((trainX_f, trainY_f), (testX_f, testY_f)) = split_f
trainX_f = trainX_f.astype("float") / 255.0         #échantillon d'apprentissage
testX_f = testX_f.astype("float") / 255.0           #échantillon test

##Transformer les noms de classes en vecteur
#Les noms de classe sont représentés par des entiers (ex: classe 'cat' = 1)
#On les transforme en vecteur (ex: classe 'cat' = [1 0 0 0 ...])
lb = LabelBinarizer()
trainY_c = lb.fit_transform(trainY_c)
testY_c = lb.transform(testY_c)
print(trainY_c) 

##↨ Architecture du réseau 3072-1024-512-20
model = Sequential()
#Première couche: 3072 neurones, fonction d'activation: sigmoïde
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
#Deuxième couche: 1024 neurones, fonction d'activation: sigmoïde
model.add(Dense(512, activation="sigmoid"))
#Troisième couche: 512 neurones, fonction d'activation: softmax
model.add(Dense(len(lb.classes_), activation="softmax"))         #ici len(lb.classes_)=20 car 20 superclasses


