###titre Entrainement d'un bras robotique grâce à un apprentissage par renforcement profond
##titre Objectifs
Ce projet a pour but de piloter un bras robotique grâce à un réseau de neurones afin de lui faire atteindre une cible définie (point rouge). La cible est générée à une position aléatoire.

##titre Conception du bras robotique
Le bras robotique est synthétisé sur un canevas tkinter. Il est composé de trois articulations. Les angles des moteurs sont définis en degrés et tournent dans le sens inverse des aiguilles d'une montre.

Les capteurs du robot sont :

*3 Rotatiomètres sur chaque moteur
*3 Capteurs de positions sur chaque moteur (ils récupèrent une valeur x et y de la distance avec la cible)
*Les angles des moteurs 3 et 2 varient entre 0 et 90 degrés par rapport à un angle de base défini. Le moteur 1 a une plage de 180 degrés.

Les mouvements possibles du robot sont définis dans "action_moteur" et permettent de faire tourner chaque moteur, dans la mesure où ce mouvement est possible (limites du canevas, plage maximale des moteurs).

Un couple est appliqué au moteur pour accélérer la rotation et afin de se rapprocher de la cible plus rapidement. Il dépend de la distance du bras avec la cible. L'unité de rotation est de 1, et le couple varie de 1 à 5.

##titre Choix de l'action
L'action est choisie via un réseau de neurones qui évalue la meilleure stratégie à adopter.

Le réseau de neurones prend en entrée la position absolue des angles des moteurs ainsi que la position relative de la cible par rapport aux moteurs, soit 9 arguments :

*3 positions d'angles de moteur
*3 * (2 coordonnées de moteurs).
En sortie, le réseau évalue la Q-value pour chacune des actions et choisit l'action qui maximise la Q-value.

La Q-value est calculée à travers l'équation de Bellman. Comme l'objectif est d'atteindre la cible en 20 étapes, le gamma pour définir la q_value target (cf équation de Bellman) est fixé à 0,9 (0,9^20 = 0,12 ~ 0).

##titre Apprentissage
#titre Déroulement de l'entrainement
L'entraînement s'effectue en 2 étapes : collecte des données, et entraînement du réseau de neurones.

Pour entraîner le bras, 100 cycles sont effectués, chacun composé de 200 étapes maximum. La position initiale, la récompense, la position finale et le mouvement sélectionné sont conservés et enregistrés et mélangés dans un dataframe pandas. Ces données sont réutilisées pour ré-entraîner le bras.

Un cycle s'arrête lorsque le bras atteint la cible ou lorsque 200 étapes sont écoulées.

Après chaque cycle, le bras est réinitialisé aléatoirement.

Au terme de 10 cycles, le réseau de neurones est entraîné sur les données collectées.


#titre Déplacement
Durant l'apprentissage, deux modes de déplacement sont utilisés : l'exploration et l'exploitation. Le choix entre les deux modes est aléatoire suivant la loi de probabilité suivante :

Au cycle 1, 100 % des choix sont exploratoires.
Au cycle n, on suit une loi récursive : la probabilité d'exploration au cycle n+1 est de 0,95 fois celle du cycle n.
Un minimum de 10 % de probabilité d'exploration est conservé.
En cas d'exploration, le mouvement est choisi aléatoirement. En cas d'exploitation, une action est prédite via le réseau de neurones et effectuée pour maximiser la Q-Value.

Titre : Le réseau de neurones
Il est composé d'une seule couche de 80 neurones cachés.
Toutes les données sont normalisées

#titre La fonction de perte
La fonction de perte est personnalisée car le gradient ne doit se propager qu'à travers la Q-Value de l'action sélectionnée pour calculer la Q-target. Ce filtre est défini via la variable de classe "mask_tensor". L'erreur est calculée comme suit :

erreur = (prédiction - cible)^2

avec Q-target = gamma * prédiction(n+1) + récompense.

##titre Limites du modèle
Ce modèle simpliste ne permet pas de définir une stratégie globale pour atteindre la cible. Ceci est problématique car, dans un mode de déplacement en rotation, la distance la plus courte n'est pas la ligne droite. Le bras reste donc bloqué sur des minima globaux (en cas de rotation qui atteint le bord ou si la cible se retrouve dans l'axe du bras). Dans ces deux cas, le bras oscille.



