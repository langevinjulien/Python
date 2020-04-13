import numpy as np
import pandas as pd                                 #Bibliothèque utilisée pour la manipulation de données
import tensorflow as tf
import imblearn                                     #Package pour l'undersampling, l'oversampling et le SMOTE 
from imblearn.over_sampling import SMOTE
from collections import Counter  
from sklearn.preprocessing import LabelBinarizer    #Pour utiliser la fonction LabelBinarizer (ligne 23)
from keras.models import Sequential                 
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D        
from keras.optimizers import SGD                    #Pour utiliser l'algorithme de descente du gradient
from sklearn.metrics import classification_report   #Pour l'évaluation du modèle
from sklearn.metrics import confusion_matrix        #Matrice de confusion
import matplotlib.pyplot as plt                     #Pour les graphiques
from keras.models import load_model                 #Pour utiliser un modèle sauvegardé
import cv2                                          #Pour importer une image





def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


meta = unpickle('C:/CIFAR100/meta')
train = unpickle('C:/CIFAR100/train')
test = unpickle('C:/CIFAR100/test')

fine_label_names=[t.decode('utf8') for t in meta[b'fine_label_names']]

print(fine_label_names.index('boy'))
print(fine_label_names.index('girl'))
print(fine_label_names.index('man'))
print(fine_label_names.index('woman'))
print(fine_label_names.index('baby'))




train_X = train[b"data"]
train_Y = train[b"fine_labels"]

test_X = test[b"data"]
test_Y = test[b"fine_labels"]





                                            #RECODAGE DES CLASSES
#Pour l'échantillon d'apprentissage
new_var=[]
for i in range(50000):
    if train_Y[i] in (11,35,46,98,2) :
        nom=0
    else:
        nom=1
    new_var.append(nom)  

#Pour l'échantillon test
new_test_Y=[]
for i in range(10000):
    if test_Y[i] in (11,35,46,98,2) :
        nom=0
    else:
        nom=1
    new_test_Y.append(nom) 

#Vérification
new_var.count(0)
new_var.count(1)





#La base de données devient alors déséquilibrée puisqu'on a une prépondérance de non-humains. 
#On choisit de faire de l'undersampling, de l'oversampling ou du smote.
#Nous avons testé les différentes manières de gérer le déséquilibre et compte tenu des résultats,
#nous avons choisi de faire du SMOTE.

                                         #SMOTE
smote = SMOTE(sampling_strategy=0.25)
X_smote, Y_smote = smote.fit_resample(train_X, new_var)
#Résumé de la distribution des classes
print(Counter(Y_smote)) 





##Transformation des noms de classes en vecteur
#Les noms de classe sont représentés par des entiers (ex: classe 'humain' = 0)
#On les transforme en vecteur (ex: classe 'cat' = [1 0 0 0 ...])
lb = LabelBinarizer()
Y_smote = lb.fit_transform(Y_smote)
new_test_Y= lb.transform(new_test_Y)

train_Y=np.array([[1,0] if l==1 else [0,1] for l in Y_smote])
test_Y=np.array([[1,0] if l==1 else [0,1] for l in new_test_Y])





##Normalisation des données
train_X = X_smote.reshape(59375, 3072).astype('float32') 
test_X = test_X.reshape(10000, 3072).astype('float32')

train_X /= 255.0
test_X /= 255.0



#Arborescence du réseau de neurones
model = Sequential()
model.add(Dense(3072, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(len(lb.classes_), activation='softmax'))

#Nombre d'itérations
epochs = 100

##Compilation du modèle
model.compile(loss="binary_crossentropy", optimizer="adam", #fonction de perte
              metrics=["accuracy"]) 

##Entraînement du modèle 
H = model.fit(train_X, train_Y, epochs=epochs, batch_size=64, verbose=1, validation_split=0.2)
#A essayer avec des batch size moins élevés
score = model.evaluate(test_X, test_Y, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


##Evaluation du modèle
liste_classes = ["Non-humain","Humain"]
print("Evaluation du modèle")
predictions = model.predict(test_X,batch_size=32)
print(classification_report(test_Y.argmax(axis=1), predictions.argmax(axis=1), target_names=liste_classes))


N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()





#A REGARDER SI CA FONCTIONNE APRES : J'ai pas du tout touché à cette partie



#Sauvegarde du modèle
model.save("C:/Users/Utilisateur/Documents/M2_SEMESTRE_2/4_Python/Projet/projet.h5")
print("Modèle sauvegardé")


##Test du modèle sur d'autres images
#Chargement du modèle
model = load_model('C:/Users/Utilisateur/Documents/M2_SEMESTRE_2/4_Python/Projet/projet.h5')

#Importation de l'image
image = cv2.imread("C:/Users/Utilisateur/Documents/M2_SEMESTRE_2/4_Python/Projet/images/image1.jpg") #Importation
image = cv2.resize(image, (32, 32))     #Changement des dimensions (pour qu'elles soient identiques à celles des images du modèle)

#Transformation des pixels cela en float entre 0 et 1
image = image.astype("float") / 255.0
image = image.reshape((1, 3072))

#Prédiction de l'image
preds = model.predict(image)

#Graphique de la probabilité d'appartenance à chacune des catégories
plt.figure(figsize = [10,5])  

x = ["Humain","Non-humain"]
y = [ preds[0][0], preds[0][1] ]
#Avec preds[0][0] = probabilité d'être un garçon
#preds[0][0] = probabilité d'être un humain
#preds[0][1] = probabilité d'être un non-humain


#Graphique du résultat (exécuter toutes les lignes qui suivent en une seule fois)
plt.barh(x, y, color='grey')

ticks_x = np.linspace(0, 1, 11)   # (start, end, number of ticks)
plt.xticks(ticks_x, fontsize=10, family='fantasy', color='black')
plt.yticks( size=15, color='navy' )
for i, v in enumerate(y):
    plt.text(v, i, "  "+str((v*100).round(1))+"%", color='blue', va='center', fontweight='bold')

plt.title('Probablité prédite', family='serif', fontsize=15, style='italic', weight='bold', color='red', loc='center', rotation=0)
plt.xlabel('Probabilité', fontsize=12, weight='bold', color='blue')
plt.ylabel('Catégorie', fontsize=12, weight='bold', color='navy')
