from tkinter import *
from environnement import Environnement
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import math



class SubBrasRobot():

    '''

    La classe subBrasRobot représente un sous bras robotique composé d'un moteur et d'un bras. Elle permet de dessiner le
    bras robotique sur un canvas en tkinter et de le faire bouger en modifiant la valeur de son angle.

    '''


    def __init__(self, canvas, x_moteur=0, y_moteur=0, theta_base=0, longueur=100, angle_initial=0, couleur='blue'):
        # Initialisation des variables d'instance
        self.x_moteur = x_moteur  # coordonnée x du moteur
        self.y_moteur = y_moteur  # coordonnée y du moteur
        self.theta_base = theta_base  # angle  de  base du bras : theta_base <= angle possible <=  theta_base + 180 , [degres]
        self.angle_ouverture = angle_initial  # angle autorise par rapport a theta base
        self.longueur = longueur  # longueur du bras
        self.canvas = canvas  # canvas sur lequel dessiner le bras
        self.couleur_moteur = couleur  # couleur du moteur
        self.couleur_bras = 'grey' # couleur du bras



        # Initialisation des constantes
        self.rayon_moteur = 20  # rayon du moteur
        self.largeur_bras = 4  # largeur du bras

        # Création des objets graphiques
        self.bras = self.canvas.create_polygon(
            self.get_coordonnee_polygone(x_moteur, y_moteur, self.theta_base + self.angle_ouverture),
            fill=self.couleur_bras
        )  # dessin du bras

        # initialise coordonnées x, y du point d'accroche
        _, _, _, _,  self._x_accroche, self._y_accroche, _, _, _, _ = self.get_coordonnee_polygone(x_moteur,
                                                                                                  y_moteur,
                                                                                                  self.theta_base + self.angle_ouverture)

        self.moteur = self.canvas.create_oval(
            self.x_moteur - self.rayon_moteur,
            self.y_moteur - self.rayon_moteur,
            self.x_moteur + self.rayon_moteur,
            self.y_moteur + self.rayon_moteur,
            fill=self.couleur_moteur
        )  # dessin du moteur

        self.label_moteur = self.canvas.create_text(
            self.x_moteur,
            self.y_moteur,
            anchor="center",
            text='M',
            font=("Arial", self.rayon_moteur)
        )  # dessin de l'étiquette "M" du moteur

        self.angle_moteur = self.canvas.create_text(
            self.x_moteur,
            self.y_moteur + 1.8 * self.rayon_moteur,
            anchor="center",
            text=str(self.angle_ouverture) + "°",
            fill='red',
            font=("Arial", int(0.7 * self.rayon_moteur))
        )  # dessin de l'angle actuel du moteur



    def get_coordonnee_polygone(self, x_base, y_base, theta):
        """
        Cette fonction calcule les coordonnées des sommets d'un polygone représentant le bras robotique,
        en fonction de la position de sa base (x_base, y_base) et de l'angle theta de l'axe du bras
        par rapport à l'horizontale.

        Args:
        - x_base (float) : abscisse de la base du bras robotique
        - y_base (float) : ordonnée de la base du bras robotique
        - theta (float) : angle en degrés de l'axe du bras robotique par rapport à l'horizontale

        Returns:
        - list (float) : une liste de 8 éléments, représentant les coordonnées des 4 sommets du polygone
                         dans l'ordre suivant : (x1, y1), (x2, y2), (x_accroche, y_accroche), (x3, y3), (x4, y4)
        """

        # Conversion de l'angle theta en radians
        theta = theta / 180 * math.pi

        # Calcul des coordonnées du premier sommet (x1, y1)
        angle1 = math.pi / 2 - theta
        delta_x1 = math.cos(angle1) * self.largeur_bras / 2
        delta_y1 = math.sin(angle1) * self.largeur_bras / 2
        x1, y1 = x_base + delta_x1, y_base + delta_y1

        # Calcul des coordonnées du deuxième sommet (x2, y2)
        delta_x2 = math.cos(theta) * self.longueur
        delta_y2 = math.sin(theta) * self.longueur
        x2, y2 = x1 + delta_x2, y1 + delta_y2

        # Calcul des coordonnées du point d'accroche (x_accroche, y_accroche)
        x_accroche = x2 - delta_x1
        y_accroche = y2 - delta_y1

        # Calcul des coordonnées du troisième sommet (x3, y3)
        x3 = x2 - 2 * delta_x1
        y3 = y2 - 2 * delta_y1

        # Calcul des coordonnées du quatrième sommet (x4, y4)
        x4 = x_base - delta_x1
        y4 = y_base - delta_y1

        # Retourne la liste des coordonnées des sommets
        return [x1, y1, x2, y2, x_accroche, y_accroche, x3, y3, x4, y4]

    def delete(self):
        """
        Cette fonction supprime les éléments graphiques associés à l'objet Moteur, tels que le moteur,
        l'étiquette du moteur, l'angle du moteur et le bras.

        Paramètres:
            Aucun

        Retourne:
            Aucun
        """
        # Suppression de l'ancien moteur
        self.canvas.delete(self.moteur)

        # Suppression de l'ancienne étiquette du moteur
        self.canvas.delete(self.label_moteur)

        # Suppression de l'ancien angle du moteur
        self.canvas.delete(self.angle_moteur)

        # Suppression de l'ancien bras
        self.canvas.delete(self.bras)

    def get_x_moteur(self):
        # Renvoie la coordonnée x actuelle du moteur
        return self.x_moteur

    def set_x_moteur(self, new_x_moteur):
        # Met à jour la coordonnée x du moteur et le déplace à la nouvelle position
        self.translation(new_x_moteur, self.y)
        # Récupération de la nouvelle position de l'accroche
        _, _, _, _, self._x_accroche, _, _, _, _, _ = self.get_coordonnee_polygone(self.x,
                                                                                  self.y,
                                                                           self.theta)

    def get_y_moteur(self):
        # Renvoie la coordonnée y actuelle du moteur
        return self.y_moteur

    def set_y_moteur(self, new_y_moteur):
        # Met à jour la coordonnée y du moteur et le déplace à la nouvelle position
        self.translation(self.x, new_y_moteur)
        # Récupération de la nouvelle position de l'accroche
        _, _, _, _, _, self._y_accroche, _, _, _, _ = self.get_coordonnee_polygone(self.x,
                                                                                  self.y,
                                                                                  self.theta)

    def get_x_accroche(self):
        # Renvoie la coordonnée x actuelle de l'accroche du sous-bras
        return self._x_accroche

    def set_x_accroche(self, _):
        # impossible de modifier le point d'accroche, n'est modifie que en interne
        print("Le point d'accroche ne peut etre modifie.")
        return

    def get_y_accroche(self):
        # Renvoie la coordonnée y actuelle de l'accroche du sous-bras
        return self._y_accroche

    def set_y_accroche(self, _):
        # impossible de modifier le point d'accroche, n'est modifie que en interne
        print("Le point d'accroche ne peut etre modifie.")
        return

    def get_theta(self):
        # Renvoie l'angle actuelle'
        return self.theta

    def set_theta(self, new_theta):
        # Appelle la méthode rotation() pour faire tourner le bras vers le nouvel angle.
        self.rotation(new_theta)
        # Met à jour les coordonnées du point d'accrochage du bras.
        _, _, _, _, self._x_accroche, self._y_accroche, _, _, _, _ = self.get_coordonnee_polygone(self.x_moteur,
                                                                                  self.y_moteur,
                                                                                  self.theta)



    # Définition de la propriété x et y pour faciliter l'accès aux coordonnées du moteur
    x = property(get_x_moteur, set_x_moteur)
    y = property(get_y_moteur, set_y_moteur)
    x_accroche = property(get_x_accroche, set_x_accroche)
    y_accroche = property(get_y_accroche, set_y_accroche)

    # Définition de la propriété 'angle' pour simplifier la modification de l'angle.
    angle = property(get_theta, set_theta)



if __name__ == '__main__':
    tk = Tk()
    env = Environnement(tk)
    sub_bras = SubBrasRobot(env.canvas)
