from keras.datasets import cifar100                 #pour importer la base cifar100
import pandas                                       #bibliothèque utilisée pour la manipulation de données 
from sklearn.preprocessing import LabelBinarizer    #pour utiliser la fonction LabelBinarizer (ligne 23)
from keras.models import Sequential                 #ligne 31
from keras.layers.core import Dense, Flatten        #ligne 33
from keras.optimizers import SGD                    #Pour utiliser l'algorithme de descente du gradient
import numpy as np 


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

#Nom des superclasses
Names = ["aquatic mammals", "fish", "flowers", "food containers",
         "fruit and vegetables", "household electrical devices",
         "household furniture", "insects", "large carnivores",
         "large man-made outdoor things", "large natural outdoor scenes",
         "large omnivires and herbivores","medium-sized mammals",
         "non-insect invertebrates", "people", "reptiles", "small mammals",
         "trees", "vehicules 1", "vehicles 2"]

##Transformer les noms de classes en vecteur
#Les noms de classe sont représentés par des entiers (ex: classe 'cat' = 1)
#On les transforme en vecteur (ex: classe 'cat' = [1 0 0 0 ...])
lb = LabelBinarizer()
trainY_c = lb.fit_transform(trainY_c)
testY_c = lb.transform(testY_c)
print(trainY_c) 

##↨ Architecture du réseau 3072-1024-512-20
model = Sequential()
#Première couche: 3072 neurones (32x32x3), fonction d'activation: sigmoïde
model.add(Dense(1024, input_shape=(32,32,3), activation="sigmoid"))
#Deuxième couche: 1024 neurones, fonction d'activation: sigmoïde
model.add(Dense(512, activation="sigmoid"))
model.add(Flatten())
#Troisième couche: 512 neurones, fonction d'activation: softmax
model.add(Dense(len(lb.classes_), activation="softmax"))         #ici len(lb.classes_)=20 car 20 superclasses

#initialisation du taux d'apprentissage
init_tapp = 0.01

#Nombre d'itérations
epochs = 30

##Compilation du modèle
opt = SGD(lr=init_tapp) #SGD = Stochastic Gradient Descent
model.compile(loss="categorical_crossentropy", optimizer=opt, #fonction de perte
              metrics=["accuracy"]) 

##Entraînement du modèle (prend du temps /!\)
H = model.fit(trainX_c, trainY_c, validation_data=(testX_c, testY_c), 
              epochs=epochs, batch_size=32)

##Evaluation du modèle






















