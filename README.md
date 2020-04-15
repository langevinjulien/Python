<div style="text-align: justify">
Nous utilisons un réseau neuronal convolutif (CNN pour Convolutional
Neural Networks en anglais). Il s’agit d’un type de réseau très utilisé
pour la reconnaissance d’image.

Présentation
------------

Nous disposons d’une base de données contenant 60 000 images. Ces images
appartiennent à des grandes catégories qui sont subdivisées en
sous-catégories. Notre but est de différencier les humains des
non-humains. Nous avons donc dans notre base de données 3 000 images
d’humains contre 57 000 images qui ne contiennent pas d’humains.

En téléchargeant la base de données, nous constatons que les
échantillons d’apprentissage et de test ont déjà été créés. Notre
échantillon d’apprentissage contient donc 2 500 images d’humains et 47
500 images de non-humains. L’échantillon test est, quant à lui, composé
de 500 images d’humains et 9 500 images de non-humains.

Nous constatons donc que notre **échantillon d’apprentissage** est
**déséquilibré** puisque nous avons seulement 5% de nos images qui
contiennent des humains. Si nous effectuons un modèle sur la
reconnaissance d’images à partir de cette base, notre modèle prédira que
presque l’entièreté des images ne contiennent pas d’humains : il
détectera bien les non-humains mais pas les humains qui sont en trop
grosse minorité.

Le but est donc **d’équilibrer la présence des humains et des
non-humains**. Nous procédons à un ***SMOTE (Synthetic Minority
Over-sampling Technique)*** qui fonctionne en créant des **observations
synthétiques** fondées sur les observations minoritaires existantes.
Pour la classe minoritaire des humains, le SMOTE calcule les k plus
proches voisins. Selon la quantité de sur-échantillonnage nécessaire, un
ou plusieurs des k voisins les plus proches sont sélectionnés pour créer
les exemples synthétiques.

Ainsi notre **nouvelle base est composée à 20% d’humains et 80% de
non-humains**. Nous avons donc 11875 images d’humains et 47500 images de
non-humains dans notre ensemble d’apprentissage.

Fonctionnement du réseau neuronal
---------------------------------

Chaque image est représentée par un **array à 3 dimensions**, chacune
correspondant à l’une des couleurs primaires : rouge, bleu et vert. La
dimension de l’array est égal à hauteur \* largeur \* 3, soit dans notre
cas 32 \* 32 \* 3 = 3072.  
Ainsi, **la première couche sera composée de 3072 neurones**
c’est-à-dire **un neurone par pixel**.

A chaque couche, plusieurs neurones traitent une partie de l’image et
renvoient un neurone en sortie qui définira avec les autres neurones de
sortie une version réduite de l’image initiale. Le processus est réitéré
plusieurs fois dans le but d’arriver à la fin du réseau avec un neurone
par choix possible. Dans notre cas, un neurone correspondra au choix
“Humain” et un autre au choix “Non-humain”.

Pour le reste de l’architecture du réseau, nous avons choisi de
**réduire par 2** le nombre de neurones en input à chaque étape. La
fonction d’activation traditionnellement utilisée dans les couches
cachées pour ce type de réseau est la fonction ReLu.

Résulats
--------

</div>
