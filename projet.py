import pandas
from keras.datasets import cifar100


##Chargement des données à partir de la bibliothèque Keras
print("Chargement des données CIFAR-100...")
split = cifar100.load_data()
((trainX, trainY), (testX, testY)) = split
trainX = trainX.astype("float") / 255.0    #échantillon d'apprentissage
testX = testX.astype("float") / 255.0      #échantillon test


