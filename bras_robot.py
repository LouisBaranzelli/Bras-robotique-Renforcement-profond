

from action_moteur import Action_moteur


import random

from sub_bras_robot import SubBrasRobot
from environnement import Environnement
from tkinter import *
import random
import numpy as np


import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



import math


class BrasRobot():

    def __init__(self, environnement, x_robot, y_robot, angle_moteur_1=None, theta_base_moteur_1=None, angle_moteur_2=None,theta_base_moteur_2=None, angle_moteur_3=None, theta_base_moteur_3=None):


        self.precision = 0.96
        self.canvas = environnement.canvas
        self.environnement = environnement
        self.list_boutton = environnement.list_boutton
        self.x_robot = x_robot
        self.y_robot = y_robot
        self.ancien_reward = 0
        self.range_max_m1 = 180
        self.range_max_m2 = 90
        self.range_max_m3 = 90




        bras_robot = self.create_bras_robot(x_base=self.x_robot,
                                            y_base=self.y_robot,
                                            theta_base_moteur_1=theta_base_moteur_1,
                                            theta_base_moteur_2=theta_base_moteur_2,
                                            theta_base_moteur_3=theta_base_moteur_3,
                                            angle_moteur_1=angle_moteur_1,
                                            angle_moteur_2=angle_moteur_2,
                                            angle_moteur_3=angle_moteur_3)

        self.bras_1, self.bras_2, self.bras_3 = bras_robot



    def reset(self):


        bras_robot = -1

        self.environnement.set_target(x=random.randrange(0,700), y=random.randrange(0,700))

        while bras_robot == -1:

            self.bras_1.delete()
            self.bras_2.delete()
            self.bras_3.delete()

            theta_base_moteur_1 = -90
            theta_base_moteur_2 = -90
            theta_base_moteur_3 = -90

            angle_moteur_1 = random.randrange(0,self.range_max_m1)
            angle_moteur_2 = random.randrange(0,self.range_max_m2)
            angle_moteur_3 = random.randrange(0,self.range_max_m3)


            bras_robot = self.create_bras_robot(x_base=self.x_robot,
                                                y_base=self.y_robot,
                                                theta_base_moteur_1=theta_base_moteur_1 ,
                                                theta_base_moteur_2=theta_base_moteur_2,
                                                theta_base_moteur_3=theta_base_moteur_3,
                                                angle_moteur_1=angle_moteur_1,
                                                angle_moteur_2=angle_moteur_2,
                                                angle_moteur_3=angle_moteur_3)

            if bras_robot != -1:
                self.bras_1, self.bras_2, self.bras_3 = bras_robot

            self.ancien_reward = 0
            self.get_reward()  # initialise self.ancien_reward a la position initiale du bras

    def create_bras_robot(self, x_base, y_base, angle_moteur_1=None, theta_base_moteur_1=None, angle_moteur_2=None,theta_base_moteur_2=None, angle_moteur_3=None, theta_base_moteur_3=None, longueur_bras_1=None, longueur_bras_2=None, longueur_bras_3=None, explication=False):

        """
            Crée un bras robotique avec 3 articulations qui peut être affiché sur un canevas tkinter.
            Les paramètres optionnels définissent la longueur des bras et l'angle des moteurs. Les angles des moteurs sont
            définis en degrés. 0 degres = 3h, 270 degres = 12h, 180 degres = 9h.
            L'angle relatif au zero trigonometrique (dans le sens oppose au cercle trigomometrique) est defini par :
            son angle d'ouverture
            son angle 0 (theta_base) correspondant a l'angle entre la position de base du bras du robot quand l'angle
            d'ouverture et a 0 et l'angle 0 du cercle trigonometrique

            Args:
                x_base (int): la coordonnée x de la base du bras
                y_base (int): la coordonnée y de la base du bras
                angle_moteur_1 (float, optional): l'angle du moteur 1 en degrés. Par défaut, l'angle est de 90 degrés.
                theta_base_moteur_1 (float, optional): l'angle de la base du moteur 1 en degrés. Par défaut, l'angle est de
                270 degrés.
                angle_moteur_2 (float, optional): l'angle du moteur 2 en degrés. Par défaut, l'angle est de 90 degrés.
                theta_base_moteur_2 (float, optional): l'angle de la base du moteur 2 en degrés. Par défaut, l'angle est de
                180 degrés.
                angle_moteur_3 (float, optional): l'angle du moteur 3 en degrés. Par défaut, l'angle est de 90 degrés.
                theta_base_moteur_3 (float, optional): l'angle de la base du moteur 3 en degrés. Par défaut, l'angle est de
                270 degrés.
                longueur_bras_1 (float, optional): la longueur du bras 1 en pixels. Par défaut, la longueur est de 500 pixels.
                longueur_bras_2 (float, optional): la longueur du bras 2 en pixels. Par défaut, la longueur est de 300 pixels.
                longueur_bras_3 (float, optional): la longueur du bras 3 en pixels. Par défaut, la longueur est de 300 pixels.

            Returns:
                int: -1 si une erreur s'est produite, sinon retourne les sous bras cres

            Raises:
                ValueError: si l'un des paramètres `angle_moteur_*` ou `theta_base_moteur_*` n'est pas un nombre ou est hors des
                limites autorisées (0 <= angle_moteur_* <= 180 et 0 <= theta_base_moteur_* <= 360).

            Warnings:
                UserWarning: si l'un des points du bras sort du canevas ou si l'un des angles des moteurs est hors des limites
                autorisées.

            """


        # Ajout des valeurs par defaut
        if longueur_bras_1 == None:
            longueur_bras_1 = 350

        if longueur_bras_2 == None:
            longueur_bras_2 = 200

        if longueur_bras_3 == None:
            longueur_bras_3 = 350

        if angle_moteur_1 == None:
            angle_moteur_1 = 0
        if theta_base_moteur_1 == None:
            theta_base_moteur_1 = -90

        if angle_moteur_2 == None:
            angle_moteur_2 = 0
        if theta_base_moteur_2 == None:
            theta_base_moteur_2 = -45

        if angle_moteur_3 == None:
            angle_moteur_3 = 0
        if theta_base_moteur_3 == None:
            theta_base_moteur_3 = 45

        largeur = int(self.canvas.__getitem__('width'))
        hauteur = int(self.canvas.__getitem__('height'))

        # conversion des angles en radian
        # le sens trignometrique est inverse au sens des moteur ce qui justifie le -
        angle_abs_moteur_1_rad = (math.pi / 180) * (theta_base_moteur_1 + angle_moteur_1)
        angle_abs_moteur_2_rad = (math.pi / 180) * (theta_base_moteur_2 + angle_moteur_2)
        angle_abs_moteur_3_rad = (math.pi / 180) * (theta_base_moteur_3 + angle_moteur_3)

        # calcule les coordonnees de chaque point du bras en cas de ratation du moteur 1
        x1_final = x_base
        x2_final = x1_final + math.cos(angle_abs_moteur_1_rad) * longueur_bras_1
        x3_final = x2_final + math.cos(angle_abs_moteur_2_rad) * longueur_bras_2
        x3_extremite = x3_final + math.cos(angle_abs_moteur_3_rad) * longueur_bras_3


        y1_final = y_base
        y2_final = y1_final + math.sin(angle_abs_moteur_1_rad) * longueur_bras_1
        y3_final = y2_final + math.sin(angle_abs_moteur_2_rad) * longueur_bras_2
        y3_extremite = y3_final + math.sin(angle_abs_moteur_3_rad) * longueur_bras_3

        # verifie que aucun point du bras ne soit a l'exterieur du canvas
        if angle_moteur_1 < 0:
            if explication: print(f"Angle du moteur 1 hors range {angle_moteur_1} ")
            return -1
        if angle_moteur_1 > self.range_max_m1:
            if explication: print(f"Angle du moteur 1 hors range {angle_moteur_1} ")
            return -1
        if angle_moteur_2 < 0:
            if explication: print(f"Angle du moteur 2 hors range {angle_moteur_2} ")
            return -1
        if angle_moteur_2 > self.range_max_m2:
            if explication: print(f"Angle du moteur 2 hors range {angle_moteur_2} ")
            return -1
        if angle_moteur_3 < 0:
            if explication: print(f"Angle du moteur 3 hors range {angle_moteur_3} ")
            return -1
        if angle_moteur_3 > self.range_max_m3:
            if explication: print(f"Angle du moteur 3 hors range {angle_moteur_3} ")
            return -1

        if x1_final < 0 or x1_final > largeur:
            if explication: print(f"Bras 1 hors canevas {x1_final} ")
            return -1

        if y1_final < 0 or y1_final > hauteur:
            if explication: print(f"Bras 1 hors canevas {y1_final} ")
            return -1

        if x2_final < 0 or x2_final > largeur:
            if explication: print(f"Bras 2 hors canevas {x2_final} ")
            return -1

        if y2_final < 0 or y2_final > hauteur:
            if explication: print(f"Bras 2 hors canevas {y2_final} ")
            return -1

        if x3_final < 0 or x3_final > largeur:
            if explication: print(f"Bras 3 hors canevas {x3_final} ")
            return -1

        if y3_final < 0 or y3_final > hauteur:
            if explication: print(f"Bras 3 hors canevas {y3_final} ")
            return -1

        if x3_extremite < 0 or x3_extremite > largeur:
            if explication: print(f"Bras 3 hors canevas {x3_extremite} ")
            return -1

        if y3_extremite < 0 or y3_extremite > hauteur:
            if explication: print(f"Bras 3 hors canevas {y3_extremite} ")
            return -1

        # Si tout est ok, creation des 3 bras
        bras_1 = SubBrasRobot(self.canvas,
                              x_moteur=x_base,
                              y_moteur=y_base,
                              longueur=350,
                              theta_base=theta_base_moteur_1,
                              angle_initial=angle_moteur_1,
                              couleur='red')

        bras_2 = SubBrasRobot(self.canvas,
                              x_moteur=bras_1.x_accroche,
                              y_moteur=bras_1.y_accroche,
                              longueur=200,
                              theta_base=theta_base_moteur_2,
                              angle_initial=angle_moteur_2,
                              couleur='green')

        bras_3 = SubBrasRobot(self.canvas,
                              x_moteur=bras_2.x_accroche,
                              y_moteur=bras_2.y_accroche,
                              longueur=350,
                              theta_base=theta_base_moteur_3,
                              angle_initial=angle_moteur_3,
                              couleur='yellow')

        return bras_1, bras_2, bras_3


    def mouvement_bras(self, orientation_moteur_1=0, couple_moteur_1=1, orientation_moteur_2=0, couple_moteur_2=1,
                       orientation_moteur_3=0, couple_moteur_3=1, explication=False):
        """
        Effectue le mouvement des bras du robot en fonction des paramètres donnés. Si le bras ne peut effectuer le mouvement, retourne -1
        :param orientation_moteur_1: orientation du moteur 1 (-1, 0 ou 1)
        :param couple_moteur_1: couple appliqué sur le moteur 1
        :param orientation_moteur_2: orientation du moteur 2 (-1, 0 ou 1)
        :param couple_moteur_2: couple appliqué sur le moteur 2
        :param orientation_moteur_3: orientation du moteur 3 (-1, 0 ou 1)
        :param couple_moteur_3: couple appliqué sur le moteur 3
        :return: -1 si erreur
        """
        test_mouvement_possible = self.mouvement_is_possible(orientation_moteur_1, couple_moteur_1, orientation_moteur_2, couple_moteur_2,
                                   orientation_moteur_3, couple_moteur_3, explication)
        if test_mouvement_possible == -1:
            return -1
        else:

            # Supprime les anciens bras`
            self.bras_1.delete()
            self.bras_2.delete()
            self.bras_3.delete()

            # Reassigne les bras cres , eregistrement de toutes les variables des angles des moteurs
            self.bras_1, self.bras_2, self.bras_3 = test_mouvement_possible

            if explication: print(f"Nouvelle position du bras {self.bras_3.x_accroche} / {self.bras_3.y_accroche}.")


    def mouvement_is_possible(self, orientation_moteur_1=None, couple_moteur_1=None, orientation_moteur_2=None, couple_moteur_2=None,
                       orientation_moteur_3=None, couple_moteur_3=None, explication=False):

        """
         Vérifie si le mouvement spécifié est possible pour le robot.

         Args:
             orientation_moteur_1 (float): Orientation de rotation du moteur 1 (1, 0 , -1).
             couple_moteur_1 (float): Coefficient qui multiplie la rotation du moteur 1
             orientation_moteur_2 (float): Orientation de rotation du moteur 2 (1, 0 , -1).
             couple_moteur_2 (float): oefficient qui multiplie la rotation du moteur 2
             orientation_moteur_3 (float): Orientation de rotation du moteur 3 (1, 0 , -1).
             couple_moteur_3 (float): oefficient qui multiplie la rotation du moteur 3
             explication (bool): Affiche des informations supplémentaires si True.

         Returns:
             bras (obj): Le nouveau bras créé si le mouvement est possible, sinon -1.

         Raises:
             Aucune exception.

        """

        # Définition de l'unité de mouvement en degrés
        unit_mouvement = 1

        # Calcul de l'angle de rotation de chaque moteur en fonction des paramètres donnés
        angle_rotation_moteur_1 = unit_mouvement * orientation_moteur_1 * couple_moteur_1
        angle_rotation_moteur_2 = unit_mouvement * orientation_moteur_2 * couple_moteur_2
        angle_rotation_moteur_3 = unit_mouvement * orientation_moteur_3 * couple_moteur_3

        # Calcul des nouveaux angles d'ouverture des moteurs après le mouvement
        nouvel_angle_moteur_1 = self.bras_1.angle_ouverture + angle_rotation_moteur_1
        nouvel_angle_moteur_2 = self.bras_2.angle_ouverture + angle_rotation_moteur_2
        nouvel_angle_moteur_3 = self.bras_3.angle_ouverture + angle_rotation_moteur_3

        if nouvel_angle_moteur_1 > self.range_max_m1 or nouvel_angle_moteur_1 < 0 or \
                nouvel_angle_moteur_2 > self.range_max_m2 or nouvel_angle_moteur_2 < 0 or \
                nouvel_angle_moteur_3 > self.range_max_m3 or nouvel_angle_moteur_3 < 0:
            return -1

        try_bras_2_theta_base = self.bras_2.theta_base + angle_rotation_moteur_1
        try_bras_3_theta_base = self.bras_3.theta_base + angle_rotation_moteur_1 + angle_rotation_moteur_2

        # Récupération des paramètres nécessaires pour la création d'un nouveau bras
        x_base = self.x_robot
        y_base = self.y_robot
        theta_base_moteur_1 = self.bras_1.theta_base
        theta_base_moteur_2 = try_bras_2_theta_base
        theta_base_moteur_3 = try_bras_3_theta_base
        longueur_bras_1 = self.bras_1.longueur
        longueur_bras_2 = self.bras_2.longueur
        longueur_bras_3 = self.bras_3.longueur

        if explication: print(f"Ancienne position du bras {self.bras_3.x_accroche} / {self.bras_3.y_accroche}.")
        # Affichage des positions des angles de chaque moteur (angle de base + angle d'ouverture)
        if explication: print(
            f'Valeurs des angles absolus avant mouvement: \n Moteur 1: {theta_base_moteur_1} + {self.bras_1.angle_ouverture}'
            f' = {theta_base_moteur_1 + self.bras_1.angle_ouverture},'
            f' \n Moteur 2: {theta_base_moteur_2} + {self.bras_2.angle_ouverture}'
            f' = {theta_base_moteur_2 + self.bras_2.angle_ouverture}, '
            f'\n Moteur 3: {theta_base_moteur_3} + {self.bras_3.angle_ouverture}'
            f' = {theta_base_moteur_3 + self.bras_3.angle_ouverture}')

        # print('Theta base moteur 1 outside, balise 0:', self.bras_1.theta_base)

        # Creation du nouveau bras aux bonnes coordonnees
        bras = self.create_bras_robot(x_base=x_base,
                                      y_base=y_base,
                                      theta_base_moteur_1=theta_base_moteur_1,
                                      theta_base_moteur_2=theta_base_moteur_2,
                                      theta_base_moteur_3=theta_base_moteur_3,
                                      angle_moteur_1=nouvel_angle_moteur_1,
                                      angle_moteur_2=nouvel_angle_moteur_2,
                                      angle_moteur_3=nouvel_angle_moteur_3,
                                      longueur_bras_1=longueur_bras_1,
                                      longueur_bras_2=longueur_bras_2,
                                      longueur_bras_3=longueur_bras_3,
                                      explication=explication)

        # Si le bras n'est pas cree retourne un code d'erreur
        if bras == -1:
            # warnings.warn("Mouvement impossible")
            return -1

        return bras


    def get_distance_from_moteur(self):
        """
        Calcule la distance entre chaque moteur du bras robotique et la cible à atteindre dans l'environnement actuel.

        Returns:
            Un tuple de trois valeurs correspondant à la distance entre chaque moteur et la cible.

        Raises:
            Aucune exception n'est levée par cette fonction.

        """

        # Récupération des coordonnées de la cible et de chaque moteur du bras



        x_target = self.environnement.x_target
        y_target = self.environnement.y_target

        x_moteur_2 = self.bras_2.x_moteur
        y_moteur_2 = self.bras_2.y_moteur

        x_moteur_3 = self.bras_3.x_moteur
        y_moteur_3 = self.bras_3.y_moteur

        x_extremite_3 = self.bras_3.x_accroche
        y_extremite_3 = self.bras_3.y_accroche






        # Retourne un tuple de trois valeurs correspondant à la distance entre chaque moteur et la cible
        return (
               x_moteur_2 - x_target, \
               y_moteur_2 - y_target,\
               x_moteur_3 - x_target, \
               y_moteur_3 - y_target,\
               x_extremite_3 - x_target, \
               y_extremite_3 - y_target
            )


    def step(self, action_moteur, couple_moteur=1, explication=False):
        """
        Fonction qui effectue une étape de mouvement du bras en fonction de l'action fournie et renvoie la position
        actuelle des moteurs et la récompense correspondante.

        Args:
            action_moteur (tuple): Un tuple contenant les actions des moteurs dans l'intervalle [-1, 1].
            couple_moteur (float): La valeur du couple à appliquer à tous les moteurs. Par défaut, 1.

        Returns:
            tuple: Un tuple contenant les positions des moteurs dans l'intervalle [0, 180] et la récompense associée.
        """

        # [Pour affichage] On récupère la position ancienne de chaque moteur.
        if explication:
            position_m1 = self.bras_1.angle_ouverture
            position_m2 = self.bras_2.angle_ouverture
            position_m3 = self.bras_3.angle_ouverture
            print(
            f"Ancienne Position moteur 1: {position_m1}, moteur 2: {position_m2}, moteur 3:{position_m3}.")

        action_m1, action_m2, action_m3 = action_moteur
        couple_m1 = couple_m2 = couple_m3 = couple_moteur

        if explication: print(f"Action efectue: Moteur 1: {action_m1*couple_m1}, Moteur 2: {action_m2*couple_m2}, Moteur 3: {action_m3*couple_m3}")

        # On effectue le mouvement du bras en fonction des actions et des couples fournis.
        erreur = self.mouvement_bras(action_m1, couple_m1, action_m2, couple_m2, action_m3, couple_m3, explication=False)

        # On calcule la récompense correspondante à cette action.
        reward = self.get_reward()

        # Si le mouvement mene a une action impossible retourne donne une reward negative.
        if erreur == -1:
            reward = 0

        # On récupère la position actuelle de chaque moteur.
        position_m1 = self.bras_1.angle_ouverture
        position_m2 = self.bras_2.angle_ouverture
        position_m3 = self.bras_3.angle_ouverture

        if explication: print(f"Nouvelle Position moteur 1: {position_m1}, moteur 2: {position_m2}, moteur 3:{position_m3}.")


        # On renvoie les positions et la récompense associée sous forme de tuple.
        return (position_m1, position_m2, position_m3), reward

    def click_gauche(self, event):
        """
        Cette fonction est appelée lorsqu'un clic gauche est détecté sur la fenêtre.
        Elle parcourt la liste des boutons pour trouver le bouton cliqué et appelle
        la méthode de mouvement de bras correspondante en fonction du bouton cliqué.

        :param event: L'événement de clic qui a été détecté.
        """
        # Récupération des coordonnées de la souris lors du clic
        x, y = event.x, event.y

        # Parcours de la liste des boutons pour trouver le bouton cliqué
        for each_boutton in self.list_boutton:
            if each_boutton.est_dedans(x, y):  # Vérification si le clic est à l'intérieur du bouton
                nom_boutton = each_boutton.id  # Récupération de l'identifiant du bouton cliqué

                # Appel de la méthode de mouvement de bras correspondante en fonction du bouton cliqué
                if nom_boutton == 'M1-':
                    self.mouvement_bras(orientation_moteur_1=-1)
                if nom_boutton == 'M1+':
                    self.mouvement_bras(orientation_moteur_1=1)
                if nom_boutton == 'M2-':
                    self.mouvement_bras(orientation_moteur_2=-1)
                if nom_boutton == 'M2+':
                    self.mouvement_bras(orientation_moteur_2=1)
                if nom_boutton == 'M3-':
                    self.mouvement_bras(orientation_moteur_3=-1)
                if nom_boutton == 'M3+':
                    self.mouvement_bras(orientation_moteur_3=1)

    def get_reward(self):
        """
        Calcule la récompense associée à une configuration actuelle du bras robotique.
        La récompense est calculée en fonction de la distance entre le point d'accroche du bras et la cible, ainsi que
        de la logueur totale du bras.
        la reward est proportionnelle a la distance avec la cible et varie de (0 a 125).
        La reward depend de la reward precedente (peut etre negative si le bras s'eloigne de la cible)

        Returns:
        reward (float): la récompense calculée pour la configuration actuelle du bras.
        """

        # Calcul de la longueur totale du bras
        longueur_bras = self.bras_1.longueur + self.bras_2.longueur + self.bras_3.longueur

        # Calcul de la distance entre le point d'accroche du bras et la cible
        distance = self.get_distance_cible()

        # Calcul de la récompense en fonction de la longueur du bras et de la distance à la cible
        # La recompense evolue exponentiellement si on se rapproche de la cible
        # recompense max : 125  = 5 * 5 * 5 =  la cible est atteinte
        # recompense mini: 0
        reward = longueur_bras - distance
        reward = (reward / longueur_bras) * np.sqrt(125)
        if reward > 0:
            reward = reward * reward
        if reward < 0:
            reward = - reward * reward




        if reward > 0 :
            delta_reward = reward - self.ancien_reward
        else: delta_reward = reward
        self.ancien_reward = reward

        if self.get_cible_atteinte(precision=self.precision) == 1:
            delta_reward = 5

        # # Si la récompense est négative, on la met à 0 pour éviter les récompenses négatives
        # if delta_reward < 0:
        #     delta_reward = 0

        # mise au cube de la recompense
        return delta_reward



    def get_distance_cible(self):

        delta_x = math.fabs(self.bras_3.x_accroche - self.environnement.x_target)
        delta_y = math.fabs(self.bras_3.y_accroche - self.environnement.y_target)
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

        return distance

    def get_cible_atteinte(self, precision=0.9):
        """
        Vérifie si la cible a été atteinte par le bras robotique.
        La cible est considérée comme atteinte si la distance entre le point d'accroche du bras et la cible est inférieure
         à une certaine proportion de la longueur totale du bras.

        Args:
        precision (float): la précision à atteindre pour considérer que la cible a été atteinte. La valeur par défaut
        est de 0.9.

        Returns:
        atteinte (bool): True si la cible a été atteinte, False sinon.
        """

        # Vérification de la validité de la précision
        if precision > 1 or precision <= 0:
            raise ArithmeticError(f'Precision non valide. ({precision})')

        # Avertissement si la précision est trop basse
        if precision < 0.8:
            raise AttributeError(f'Precision inadaptee. ({precision})')

        # Calcul de la distance entre le point d'accroche du bras et la cible
        delta_x = math.fabs(self.bras_3.x_accroche - self.environnement.x_target)
        delta_y = math.fabs(self.bras_3.y_accroche - self.environnement.y_target)
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

        # Calcul de la longueur totale du bras
        longueur_bras = self.bras_1.longueur + self.bras_2.longueur + self.bras_3.longueur

        # Vérification si la cible est atteinte en comparant la distance à une proportion de la longueur totale du bras
        if distance < (1 - precision) * longueur_bras:
            return True
        else:
            return

    def get_mask_bras(self, position_moteurs: pd.DataFrame):
        """
        Fonction qui prend en entrée un dataframe contenant les angles de position des 3 moteurs d'un bras mécanique
        et renvoie un dataframe binaire où chaque ligne correspond à une position des moteurs et chaque colonne
        correspond à un angle possible pour chaqu'un des 3 moteurs.

        Args:
        position_moteurs (pd.DataFrame): dim (n, 3) Dataframe avec les angles de position des 3 moteurs [angle_bras_1, angle_bras_2, angle_bras_3] angle de: 0, 180 degres

        Returns:
        pd.DataFrame : dim (nx542)) Dataframe binaire avec 3*181 = 542 colonnes [0,1] où chaque ligne correspond à une configuration du bras.

        Raises:
        AttributeError : Si la dimension du dataframe position_moteurs est différente de n x 3
        AttributeError : Si une position de moteur est en dehors de l'intervalle [0, 180]
        """

        # Vérification de la dimension du dataframe
        _, nbr_colonne = position_moteurs.shape
        if nbr_colonne != 3:
            raise AttributeError(
                f"Position moteur doit etre un dataframe de dim n x 3 [nombre de colonne: {nbr_colonne}]")

        # Initialisation du dataframe à renvoyer
        position_moteurs_return = pd.DataFrame()

        # Parcours de chaque position de moteurs
        for i in range(len(position_moteurs)):
            # Récupération des angles de position des 3 moteurs
            angle_bras_1, angle_bras_2, angle_bras_3 = position_moteurs.iloc[i, 0], \
                                                       position_moteurs.iloc[i, 1], \
                                                       position_moteurs.iloc[i, 2]

            # Vérification que les angles sont dans l'intervalle [0, 180]
            if angle_bras_1 > self.range_max_m1 or angle_bras_1 < 0:
                raise AttributeError(f"Position du bras 1 erreur [{angle_bras_1}]")
            if angle_bras_2 > self.range_max_m2 or angle_bras_2 < 0:
                raise AttributeError(f"Position du bras 2 erreur [{angle_bras_2}]")
            if angle_bras_3 > self.range_max_m3 or angle_bras_3 < 0:
                raise AttributeError(f"Position du bras 3 erreur [{angle_bras_3}]")

            # Création de 3 tableaux numpy avec des 0 partout sauf à la colonne correspondant à l'angle de position du moteur
            array_1 = np.zeros(shape=(1, 181))
            array_2 = np.zeros(shape=(1, 181))
            array_3 = np.zeros(shape=(1, 181))

            array_1[0, angle_bras_1] = 1
            array_2[0, angle_bras_2] = 1
            array_3[0, angle_bras_3] = 1

            # Concaténation des 3 tableaux pour former une seule ligne du dataframe binaire de 542 colonnes
            nouveau_df_ligne = pd.DataFrame(np.concatenate((array_1, array_2, array_3), axis=1))

            # Concaténation de la ligne au dataframe final
            position_moteurs_return = pd.concat((position_moteurs_return, nouveau_df_ligne), axis=0)

        # Retourne le dataframe binaire
        return position_moteurs_return



if __name__ == '__main__':
    tk = Tk()
    env = Environnement(tk)
    BrasRobot(env, 50, 750)


