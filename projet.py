import pandas                                       #bibliothèque utilisée pour la manipulation de données 
from sklearn.preprocessing import LabelBinarizer    #pour utiliser la fonction LabelBinarizer (ligne 23)
from keras.models import Sequential                 
from keras.layers.core import Dense, Flatten        
from keras.optimizers import SGD                    #Pour utiliser l'algorithme de descente du gradient
import numpy as np 
from sklearn.metrics import classification_report   #Pour l'évaluation du modèle
import matplotlib.pyplot as plt                     #Pour les graphiques
from keras.models import load_model                 #Pour utiliser un modèle sauvegardé
import cv2                                          #Pour importer une image

##Importation des données
#Fonction d'importation
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

    
#Importation des échantillons train et test
train = unpickle('C:/Users/Utilisateur/Documents/M2_SEMESTRE_2/4_Python/Projet/cifar-100-python/train')
test = unpickle('C:/Users/Utilisateur/Documents/M2_SEMESTRE_2/4_Python/Projet/cifar-100-python/test')
train_X = train[b'data'] 
train_Y = train[b'fine_labels']
test_X = test[b'data']
test_Y = test[b'fine_labels']

#Importation des noms des classes
meta = unpickle('C:/Users/Utilisateur/Documents/M2_SEMESTRE_2/4_Python/Projet/cifar-100-python/meta')
fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
fine_label_names[0]
print(fine_label_names.index('boy'))   #boy est la classe numéro 11
print(fine_label_names.index('girl'))  #girl est la classe numéro 35
print(fine_label_names.index('woman')) #woman est la classe numéro 98
print(fine_label_names.index('man'))   #man est la classe numéro 46
print(fine_label_names.index('baby'))  #baby est la classe numéro 2


#Changement du nom de la classe
liste_classes = ["Garçon","Fille","Femme","Homme","Bébé","Non-humain"]
#Dans l'échantillon d'apprentissage
for i in range(50000):
    if train_Y[i]==11:
        train_Y[i]=0      #0: garçon
    elif train_Y[i]==35:
        train_Y[i]=1      #1: fille
    elif train_Y[i]==98:
        train_Y[i]=2      #2: femme
    elif train_Y[i]==46:
        train_Y[i]=3      #3: homme
    elif train_Y[i]==2:
        train_Y[i]=4      #4: bébé
    else:
        train_Y[i]=5      #5: non-humain
 
#Dans l'échantillon test       
for i in range(10000):
    if test_Y[i]==11:
        test_Y[i]=0
    elif test_Y[i]==35:
        test_Y[i]=1
    elif test_Y[i]==98:
        test_Y[i]=2
    elif test_Y[i]==46:
        test_Y[i]=3
    elif test_Y[i]==2:
        test_Y[i]=4
    else:
        test_Y[i]=5


#Récupérer la liste des éléments humains
humains = [x for x in range(len(train_Y)) if train_Y[x] != 5]
#Table des humains
c=np.array(train_Y)
d=np.array(train_X)
train_Y_humains=c[humains]
train_X_humains=d[humains]

#Récupérer la liste des éléments humains non-humains
non_humains=[i for i, n in enumerate(train_Y) if n == 5]
train_Y_non_humains=c[non_humains]
train_X_non_humains=d[non_humains]

##Undersampling
#Echantillon train de départ (50000 obs): 5% humains (2500 obs) / 95% de non-humains (47500 obs)
idx = np.random.randint(47501, size=500) 
train_X_non_humains_und=train_X_non_humains[idx,:]  #Sur les explicatives
train_Y_non_humains_und=train_Y_non_humains[idx]    #Sur la variable expliquée

trainY=np.concatenate((train_Y_non_humains_und, train_Y_humains),axis=None)
trainX=np.concatenate((np.array(train_X_non_humains_und), np.array(train_X_humains), np.array(train_X_humains)), axis=0)

##Transformation des noms de classes en vecteur
#Les noms de classe sont représentés par des entiers (ex: classe 'garçon' = 1)
#On les transforme en vecteur (ex: classe 'cat' = [1 0 0 0 ...])
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
test_Y= lb.transform(test_Y)

## Architecture du réseau 3072-1024-512-20
model = Sequential()
#Première couche: 3072 neurones (32x32x3), fonction d'activation: sigmoïde
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
#Deuxième couche: 1024 neurones, fonction d'activation: sigmoïde
model.add(Dense(512, activation="sigmoid"))
#model.add(Flatten())
#Troisième couche: 512 neurones, fonction d'activation: softmax
model.add(Dense(len(lb.classes_), activation="softmax"))         #ici len(lb.classes_)=6 car 6 classes

#initialisation du taux d'apprentissage
init_tapp = 0.01

#Nombre d'itérations
epochs = 30

##Compilation du modèle
opt = SGD(lr=init_tapp) #SGD = Stochastic Gradient Descent
model.compile(loss="categorical_crossentropy", optimizer=opt, #fonction de perte
              metrics=["accuracy"]) 

##Entraînement du modèle 
H = model.fit(trainX, trainY, validation_data=(test_X, test_Y), epochs=epochs, batch_size=32)
#A essayer avec des batch size moins élevés


##Evaluation du modèle
print("Evaluation du modèle")
predictions = model.predict(test_X, batch_size=32)
print(classification_report(test_Y.argmax(axis=1), predictions.argmax(axis=1), target_names=liste_classes))

#Représentation graphique
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

x = ["Garçon","Fille","Femme","Homme","Bébé","Non-humain"]
y = [ preds[0][0], preds[0][1], preds[0][2], preds[0][3], preds[0][4], preds[0][5] ]
#Avec preds[0][0] = probabilité d'être un garçon
#preds[0][0] = probabilité d'être un garçon
#preds[0][1] = probabilité d'être une fille
#preds[0][2] = probabilité d'être une femme
#preds[0][3] = probabilité d'être un homme
#preds[0][4] = probabilité d'être un bébé
#preds[0][5] = probabilité d'être non-humain

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





    













