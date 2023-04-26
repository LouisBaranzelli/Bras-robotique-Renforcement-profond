

from action_moteur import Action_moteur
from bras_robot import BrasRobot
import pickle

import random

from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from keras.layers import Dense, Input


import keras.optimizers
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import math
from functools import partial
from tensorflow.keras.models import load_model



class Neural_network():

    def __init__(self, bras_robot: BrasRobot):
        """
        Initialise un objet MonModele avec un bras_robot de type BrasRobot.

        Args:
            bras_robot (BrasRobot): Un objet BrasRobot utilisé pour initialiser le modèle.
        """



        # Initialiser les attributs
        self.loss = []
        self.batch_mask_action_binaire_max_q = 0
        self.scaler = StandardScaler()
        self.optimizer = keras.optimizers.Nadam(lr=0.001)
        self.liste_action = Action_moteur()
        self.bras_robot = bras_robot

        # Initialiser le modèle
        self.model = keras.models.Sequential()


        # Ajouter une couche cachée avec 50 neurones et une activation relu avec régularisation L1 et L2
        self.model.add(keras.layers.Dense(80))
        # self.model.add(keras.layers.Dense(20))

        # Ajouter une couche de dropout avec un taux de 0.2 pour réduire le surapprentissage
        # self.model.add(keras.layers.Dropout(0.2))

        # Ajouter une couche de sortie avec le nombre de neurones linéaires = au nombre d'actions possibles.
        self.model.add(keras.layers.Dense(len(self.liste_action.action_possible), activation='linear'))

        # Compiler le modèle avec la fonction de perte et l'optimiseur spécifiés
        self.model.compile(loss=self.model_loss, optimizer=self.optimizer)



    def main_train_bras_robot(self, explication=True, cycle=50, step=400):
        '''
        Cette méthode permet d'entrainer un bras robotique en utilisant une politique de type Q-Learning.

        Args:
            - explication (bool) : Si True, affiche des informations supplémentaires lors de l'entrainement (default=True)
            - cycle (int) : Nombre de cycles d'apprentissage (default=50)
            - step (int) : Nombre d'étapes maximum par cycle (default=400)
        '''

        def conversion_angle(angle):
            '''
            Cette fonction convertit un angle en degrés en un angle compris entre -180° et +180°.
            '''
            if angle >= 360:
                angle = angle - 360
            if angle > 180:
                angle = angle - 360
            return angle

        self.cycle = cycle
        self.step = step

        ratio_exploration = 1
        # Création d'un DataFrame pour stocker les données de chaque étape
        df_dataset = pd.DataFrame(columns=['position_moteur_1',
                                           'position_moteur_2',
                                           'position_moteur_3',
                                           'ancien_delta_x_moteur_2',
                                           'ancien_delta_y_moteur_2',
                                           'ancien_delta_x_moteur_3',
                                           'ancien_delta_y_moteur_3',
                                           'ancien_delta_x_pointe_moteur_3',
                                           'ancien_delta_y_pointe_moteur_3',
                                           'recompense',
                                           'nouvelle_position_moteur_1',
                                           'nouvelle_position_moteur_2',
                                           'nouvelle_position_moteur_3',
                                           'nouveau_delta_x_moteur_2',
                                           'nouveau_delta_y_moteur_2',
                                           'nouveau_delta_x_moteur_3',
                                           'nouveau_delta_y_moteur_3',
                                           'nouveau_delta_x_pointe_moteur_3',
                                           'nouveau_delta_y_pointe_moteur_3',
                                           'index_action'])

        self.result_cycle = []
        nbr_step = []

        # Boucle d'apprentissage sur le nombre de cycles choisi
        for cycle_apprentissage in range(self.cycle):
            step = 0

            if explication:
                print("Regeneration de la position du bras aleatoire")
                print(f"Ratio d'exploration du cycle: {ratio_exploration}")

            self.bras_robot.reset()

            # Boucle de déplacement du bras robotique
            while step <= self.step and not (self.bras_robot.get_cible_atteinte(self.bras_robot.precision)):
                position_moteur_n = self.bras_robot.bras_1.angle_ouverture, \
                                    self.bras_robot.bras_2.angle_ouverture, \
                                    self.bras_robot.bras_3.angle_ouverture


                # Calcul du couple moteur
                couple_moteur = math.ceil((self.bras_robot.get_distance_cible()) / 200)  # Couple de 1 a 5

                # Sélection de l'action à effectuer
                index_action = self.choisir_action(position_moteur=position_moteur_n,
                                                   ratio_exploration=ratio_exploration,
                                                   couple_applique=couple_moteur)
                action = self.liste_action.get_action(index_action)

                # Enregistrement des positions absolues des moteurs avant le mouvement
                ancienne_position_abs_moteur_1 = self.bras_robot.bras_1.angle_ouverture + self.bras_robot.bras_1.theta_base
                ancienne_position_abs_moteur_1 = conversion_angle(ancienne_position_abs_moteur_1)

                ancienne_position_abs_moteur_2 = self.bras_robot.bras_2.angle_ouverture + self.bras_robot.bras_2.theta_base
                ancienne_position_abs_moteur_2 = conversion_angle(ancienne_position_abs_moteur_2)

                ancienne_position_abs_moteur_3 = self.bras_robot.bras_3.angle_ouverture + self.bras_robot.bras_3.theta_base
                ancienne_position_abs_moteur_3 = conversion_angle(ancienne_position_abs_moteur_3)

                ancien_delta_x_m2, ancien_delta_y_m2, \
                ancien_delta_x_m3, ancien_delta_y_m3, \
                ancien_delta_x_pointe_m3, ancien_delta_y_pointe_m3 = self.bras_robot.get_distance_from_moteur()

                # Execution de l'action et obtention de la nouvelle position et du reward associé
                nouvelle_position, reward = self.bras_robot.step(action, couple_moteur=couple_moteur)

                # if explication: print("Reward:", reward)
                # if explication: print("Distance:", self.bras_robot.get_distance_cible())

                self.bras_robot.environnement.tk.update()
                if explication:
                    if self.bras_robot.get_cible_atteinte(self.bras_robot.precision):
                        print(f"Cycle {cycle_apprentissage}. Cible atteinte en {step} etapes.")
                        nbr_step.append(step)
                        self.result_cycle.append(step)
                    if step == self.step:
                        print(f"Cycle {cycle_apprentissage}. --oO Fail cycle 0o--")
                        nbr_step.append(step)
                        self.result_cycle.append(step)

                # Enregistrement des positions absolues des moteurs apres le mouvement
                nouvelle_position_abs_moteur_1 = self.bras_robot.bras_1.angle_ouverture + self.bras_robot.bras_1.theta_base
                nouvelle_position_abs_moteur_1 = conversion_angle(nouvelle_position_abs_moteur_1)

                nouvelle_position_abs_moteur_2 = self.bras_robot.bras_2.angle_ouverture + self.bras_robot.bras_2.theta_base
                nouvelle_position_abs_moteur_2 = conversion_angle(nouvelle_position_abs_moteur_2)

                nouvelle_position_abs_moteur_3 = self.bras_robot.bras_3.angle_ouverture + self.bras_robot.bras_3.theta_base
                nouvelle_position_abs_moteur_3 = conversion_angle(nouvelle_position_abs_moteur_3)


                nouveau_delta_x_m2, nouveau_delta_y_m2, \
                nouveau_delta_x_m3, nouveau_delta_y_m3, \
                nouveau_delta_x_pointe_m3, nouveau_delta_y_pointe_m3 = self.bras_robot.get_distance_from_moteur()

                # Creation de la nouvelle ligne de data
                nouvelle_ligne = pd.DataFrame(
                    {'position_moteur_1': [ancienne_position_abs_moteur_1],
                     'position_moteur_2': [ancienne_position_abs_moteur_2],
                     'position_moteur_3': [ancienne_position_abs_moteur_3],
                     'ancien_delta_x_moteur_2': [ancien_delta_x_m2],
                     'ancien_delta_y_moteur_2': [ancien_delta_y_m2],
                     'ancien_delta_x_moteur_3': [ancien_delta_x_m3],
                     'ancien_delta_y_moteur_3': [ancien_delta_y_m3],
                     'ancien_delta_x_pointe_moteur_3': [ancien_delta_x_pointe_m3],
                     'ancien_delta_y_pointe_moteur_3': [ancien_delta_y_pointe_m3],

                     'nouvelle_position_moteur_1': [nouvelle_position_abs_moteur_1],
                     'nouvelle_position_moteur_2': [nouvelle_position_abs_moteur_2],
                     'nouvelle_position_moteur_3': [nouvelle_position_abs_moteur_3],
                    'nouveau_delta_x_moteur_2': [nouveau_delta_x_m2],
                    'nouveau_delta_y_moteur_2': [nouveau_delta_y_m2],
                    'nouveau_delta_x_moteur_3': [nouveau_delta_x_m3],
                    'nouveau_delta_y_moteur_3': [nouveau_delta_y_m3],


                     'nouveau_delta_x_pointe_moteur_3': [nouveau_delta_x_pointe_m3],
                     'nouveau_delta_y_pointe_moteur_3': [nouveau_delta_y_pointe_m3],


                     'index_action': [index_action],
                     'recompense': [reward]})

                # Ajout de la nouvelle ligne de data, melange et creation d'un roulement si le dataset est trop grand.
                df_dataset = pd.concat([df_dataset, nouvelle_ligne], ignore_index=True)
                df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)

                # Standardise les entrees du modele
                self.scaler.fit(df_dataset[['position_moteur_1',
                                            'position_moteur_2',
                                            'position_moteur_3',
                                            'ancien_delta_x_moteur_2',
                                            'ancien_delta_y_moteur_2',
                                            'ancien_delta_x_moteur_3',
                                            'ancien_delta_y_moteur_3',
                                            'ancien_delta_x_pointe_moteur_3',
                                            'ancien_delta_y_pointe_moteur_3'
                                            ]].to_numpy())
                # print ("liste des position du moteur 1",df_dataset.position_moteur_1)
                # print ("moyenne position du moteur 1",df_dataset.position_moteur_1.mean())
                # print ("moyenne position du moteur 2",df_dataset.position_moteur_2.mean())
                # print ("moyenne position du moteur 3",df_dataset.position_moteur_3.mean())

                if len(df_dataset) > 10000:
                    df_dataset = df_dataset.drop(0)


                step += 1


            # Periodiquement effectue un entrainement sur le dataset
            ratio_exploration = max(0.1, 0.97 * ratio_exploration)
            if cycle_apprentissage % 10 == 0 and cycle_apprentissage > 0:

                # print(nbr_step)
                if explication: print(f"\nNombre d'etape moyenne par cycle : {np.mean(nbr_step)}. ")
                nbr_step = []
                if explication: print(f"cycle_apprentissage : {cycle_apprentissage}. \n")



                self.train_model(df_dataset[['position_moteur_1',
                                             'position_moteur_2',
                                             'position_moteur_3',
                                             'ancien_delta_x_moteur_2',
                                             'ancien_delta_y_moteur_2',
                                             'ancien_delta_x_moteur_3',
                                             'ancien_delta_y_moteur_3',
                                             'ancien_delta_x_pointe_moteur_3',
                                             'ancien_delta_y_pointe_moteur_3'
                                             ]],
                                 df_dataset[['index_action']],
                                 df_dataset[['nouvelle_position_moteur_1',
                                             'nouvelle_position_moteur_2',
                                             'nouvelle_position_moteur_3',
                                             'nouveau_delta_x_moteur_2',
                                             'nouveau_delta_y_moteur_2',
                                             'nouveau_delta_x_moteur_3',
                                             'nouveau_delta_y_moteur_3',
                                             'nouveau_delta_x_pointe_moteur_3',
                                             'nouveau_delta_y_pointe_moteur_3'

                                             ]],
                                 df_dataset[['recompense']], explication=True)

        # Au terme de l'apprentissage, enregistre le scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        self.model.save("last_model.h5")


    def test(self, name_model=None):


        self.model = load_model(name_model, compile=False)

        #
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        self.bras_robot.reset()
        compteur_action = 0

        while self.bras_robot.get_cible_atteinte(0.95) != 1:

            position_moteur_n = self.bras_robot.bras_1.angle_ouverture, \
                                self.bras_robot.bras_2.angle_ouverture, \
                                self.bras_robot.bras_3.angle_ouverture

            index_action = self.choisir_action(position_moteur=position_moteur_n,
                                               ratio_exploration=0.2,
                                               couple_applique=1)
            action = self.liste_action.get_action(index_action)

            self.bras_robot.step(action, couple_moteur=1)
            compteur_action += 1
            self.bras_robot.environnement.tk.update()
            print(compteur_action)


    def choisir_action(self, position_moteur, couple_applique=1, ratio_exploration=0):
        """
        Fonction qui choisit une action à effectuer en fonction de la position du moteur et du ratio d'exploration.
        En cas d'exploration l'action est aleatoire.
        En cas d'exploitation, utilise un reseau de neurone pour definir la position du Q-maximale.

        Args:
        - position_moteur (array-like): la position actuelle du moteur, un tableau NumPy de dimension (1, 3).
        - ratio_exploration (float): le ratio d'exploration, compris entre 0 et 1.
        - position_moteur (array-like): la position actuelle
        Returns:
        - couple_applique (int): Coefficient qui multipliera la rotation du mouvement choisi. Utilise ici pour confirmer
        que l'action choisi est valide.


        """

        # Génère un nombre aléatoire entre 0 et 100
        random_0_100 = random.randrange(0, 101)

        # Si le nombre aléatoire est inférieur à 10 * le ratio d'exploration,
        # on choisit une action aléatoire
        if random_0_100 <= 100 * ratio_exploration:
            # Choisit un indice d'action aléatoire entre 0 et 5 (inclus) 0,0,0 n'est pas valable, donc 27 - 1
            index_action = self.liste_action.get_random_index()

        # Sinon, on choisit l'action ayant la plus grande valeur Q prédite par le modèle
        else:
            # input Pour la prediction: liste de la position desmoteur puis des coordonnees des moteur vis a vis de la target.
            df_pos_moteur = pd.DataFrame([list(position_moteur + self.bras_robot.get_distance_from_moteur())])

            predictions_index_action_q_ascendant = []


            predictions = self.make_prediction(df_pos_moteur, explanation=False)
            list_prediction_action_order = predictions.tolist()[0]

            list_prediction_asc = predictions.tolist()[0]
            list_prediction_asc.sort()
            # Trouve l'indice de l'action ayant la plus grande valeur Q prédite
            # On trie en ordre croissant les index des actions en fonction de leurs Q values respectives
            # On prend le dernier mouvement, si le mouvement est impossible om prend l'avant dernier etc ...
            for each_q_value in list_prediction_asc:
                predictions_index_action_q_ascendant.append(list_prediction_action_order.index(each_q_value))


            i = len(self.liste_action.action_possible) - 1
            erreur = -1
            couple_m1 = couple_m2 = couple_m3 = couple_applique
            while erreur == -1:
                index_action = predictions_index_action_q_ascendant[i]
                rotation_moteur_1, rotation_moteur_2, rotation_moteur_3 = self.liste_action.get_action(index_action)
                erreur = self.bras_robot.mouvement_is_possible(orientation_moteur_1=rotation_moteur_1,
                                                               orientation_moteur_2=rotation_moteur_2,
                                                               orientation_moteur_3=rotation_moteur_3,
                                                               couple_moteur_1=couple_m1,
                                                               couple_moteur_2=couple_m2,
                                                               couple_moteur_3=couple_m3)
                i -= 1
            bras1_test, bras2_test, bras3_test = erreur
            # Supprime les bras crees a l'occasion du test
            bras1_test.delete()
            bras2_test.delete()
            bras3_test.delete()


        return index_action


    def make_prediction(self, df_position_moteur, explanation=False):
        """
        Cette fonction prend en entrée un dataframe contenant les positions de moteur à prédire,
        ainsi qu'un paramètre optionnel qui permet d'afficher des explications sur les opérations
        effectuées par la fonction. La fonction retourne les prédictions de position de moteur.

        Args:
        - df_position_moteur: un dataframe avec 3 colonnes représentant les positions de moteur à prédire.
        - explanation (bool): si True, affiche des explications sur les opérations effectuées.

        Returns:
        - prediction: un tableau numpy contenant les prédictions de l'action qui maximise la Q valeur.

        Raises:
        - AttributeError: si le dataframe ne contient pas exactement 3 colonnes.

        """


        nbr_ligne, nbr_colonne = df_position_moteur.shape

        # Vérification que le dataframe contient bien 3 colonnes, sinon lève une exception AttributeError.
        if nbr_colonne != 9:
            raise AttributeError(
                f"Pour effectuer une prediction, veuillez fournir 9 arguments [nombre d'argument: {nbr_colonne}]")

        # Si le paramètre explanation est True, affiche les positions de moteur fournies par l'utilisateur.
        if explanation:
            print("Position moteur scaled entre 0 et 360:", df_position_moteur.to_numpy())

        # Normalisation et mise à l'échelle des positions de moteur fournies par l'utilisateur.
        scaled_moteur_pos = self.scaler.transform(df_position_moteur.values)

        # Prédiction de la position de moteur normalisée en utilisant le modèle.
        prediction = self.model.predict(scaled_moteur_pos, verbose=False)

        if explanation:
            print("Position moteur normalisée et mise à l'échelle:", scaled_moteur_pos)

        # Retourne la prediction de l'action qui maximise la Q valeur.
        return prediction


    def train_model(self, df_position_moteur_n, df_index_action, df_position_moteur_n_plus_1, df_recompense, γ=0.9, explication=False):

        """
        Entraîne le modèle d'apprentissage par renforcement en utilisant un batch de données.

        :param df_position_moteur_n: DataFrame de la position du moteur à l'étape n.
        :type df_position_moteur_n: pandas.DataFrame
        :param df_index_action: DataFrame des index des actions possibles.
        :type df_index_action: pandas.DataFrame
        :param df_position_moteur_n_plus_1: DataFrame de la position du moteur à l'étape n+1.
        :type df_position_moteur_n_plus_1: pandas.DataFrame
        :param df_recompense: DataFrame des récompenses pour chaque action possible.
        :type df_recompense: pandas.DataFrame
        :param γ: Facteur de discount pour les récompenses futures.
        :type γ: float
        :param explication: Boolean pour afficher des informations sur l'entraînement.
        :type explication: bool
        """

        # objectif : mresoudre le probleme en moin de 25 etape. Avec un gana = 0.9 => 0.9^25 = 0.07 proche de 0
        if explication: print(f"-- entrainement du modele --")

        batch_size = 32
        nombre_ligne = len(df_position_moteur_n)

        self.loss = []
        for donnee_btach_n in range(0, nombre_ligne, batch_size):

            # if explication: print(f"Batch {int(donnee_btach_n/batch_size)} / {int(nombre_ligne / batch_size)}")

            if donnee_btach_n + batch_size > nombre_ligne:
                donnee_btach_n_final = nombre_ligne
            else:
                donnee_btach_n_final = donnee_btach_n + batch_size


            batch_df_pos_moteur_n = df_position_moteur_n.iloc[donnee_btach_n:donnee_btach_n_final, :].values
            batch_vec_pos_moteur_n = self.scaler.transform(batch_df_pos_moteur_n)
            batch_ten_pos_moteur_n = tf.convert_to_tensor(batch_vec_pos_moteur_n)

            batch_pos_moteur_f = df_position_moteur_n_plus_1.loc[donnee_btach_n:donnee_btach_n_final, ['nouvelle_position_moteur_1',
                                                                                                       'nouvelle_position_moteur_2',
                                                                                                       'nouvelle_position_moteur_3',
                                                                                                       'nouveau_delta_x_moteur_2',
                                                                                                       'nouveau_delta_y_moteur_2',
                                                                                                       'nouveau_delta_x_moteur_3',
                                                                                                       'nouveau_delta_y_moteur_3',
                                                                                                       'nouveau_delta_x_pointe_moteur_3',
                                                                                                       'nouveau_delta_y_pointe_moteur_3'
                                                                                                       ]]
            batch_q_value_f = self.make_prediction(batch_pos_moteur_f, explanation=False)
            # print("\n Prediction", batch_q_value_f)
            target = np.ones(shape=(donnee_btach_n_final - donnee_btach_n, len(self.liste_action.action_possible)))
            for i in range(donnee_btach_n_final - donnee_btach_n):
                max = np.max(batch_q_value_f[i, :])
                target[i, :] = max

            vecteur_recompense = df_recompense.to_numpy()

            # print("\n recompense", vecteur_recompense[donnee_btach_n:donnee_btach_n_final, :])

            target = vecteur_recompense[donnee_btach_n:donnee_btach_n_final, :] + γ * target
            target = target.astype(float)
            target = tf.convert_to_tensor(target)

            batch_df_action = df_index_action.iloc[donnee_btach_n:donnee_btach_n_final, :]
            self.batch_mask_action_binaire_max_q = self.liste_action.get_mask_action(batch_df_action).to_numpy()

            self.model.compile(loss=self.model_loss, optimizer=self.optimizer)

            # print("target outside", target)
            # tf.print("\n")
            # print("mask outside",  batch_df_action)

            # Je ne melange pas pour que mon masque qui est une variable globale puisse correspondre dans la fonction loss


            self.model.fit(batch_ten_pos_moteur_n, target,shuffle=False, verbose=True)

        # print("Moyene de la loss durant l'entrainement:", sum(self.loss)/len(self.loss))




    def model_loss(self, q_target, q_predict):
        """

        """

        # tf.print("target inside", q_target)
        # tf.print("\n")
        # tf.print("predict inside", q_predict)
        # tf.print("\n")

        # print(f"len q predict inside:{len(q_predict)}")
        # tf.print(f"mask: {self.batch_mask_action_binaire_max_q}")
        gradien = tf.square(q_target - q_predict)


        mask_tensor = tf.convert_to_tensor(self.batch_mask_action_binaire_max_q)
        #
        # tf.print("Nombre de donnees du masque pendant  l'entrainement ", len(mask_tensor))
        #
        # tf.print("mask inside", mask_tensor)
        mask_tensor = tf.cast(mask_tensor, dtype=tf.float32)
        # tf.print("\n")

        # tf.print('Gradient sans masque', gradien)
        gradien = tf.multiply(mask_tensor, gradien)

        # tf.print('Gradient avec masque', gradien)
        # gradien = tf.reduce_max(gradien, axis=1)
        # tf.print("gradient output", gradien)

        # tf.print(gradien)

        # tf.print('Gradient final', gradien)


        # self.loss.append(gradien)
        return gradien
