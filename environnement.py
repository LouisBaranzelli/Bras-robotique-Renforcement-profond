
from tkinter import *

class Environnement():
    """
     Cette classe représente l'environnement dans lequel évolue le bras robotique.

     Attributes:
         canvas (Canvas): le canvas sur lequel les éléments de l'environnement sont dessinés.
         x_target (int): la coordonnée x de la cible que doit attraper le robot.
         y_target (int): la coordonnée y de la cibleque doit attraper le robot.
         diametre_objet (int): le diamètre de la cible que doit attraper le robot.
         list_boutton (list): la liste de tous les boutons sur le canvas.

     Methods:
         __init__(self, tk):
             Initialise l'objet Environnement avec un canvas, une cible et des boutons.
         set_target(self):
             Génère une nouvelle cible aléatoire ou fixee.
         class Boutton():
             Représente un bouton sur le canvas.
             __init__(self, canvas, x_position_centre, y_position_centre, text=""):
                 Initialise un objet Boutton avec des informations sur sa position et son aspect graphique.
             est_dedans(self, x_target, y_target):
                 Vérifie si un point est situé à l'intérieur du bouton.
     """

    def __init__(self, tk):
        # Création d'un canvas

        self.tk = tk
        self.canvas = Canvas(tk, width=1000, height=1000)
        self.canvas.pack()

        # Récupération de la taille du canvas
        self.largeur = int(self.canvas.__getitem__('width'))
        self.hauteur = int(self.canvas.__getitem__('height'))

        # Initialisation de la cible
        self.x_target = 0
        self.y_target = 0
        self.diametre_objet = 10
        self.target = self.canvas.create_oval(self.x_target - self.diametre_objet,
                                self.y_target - self.diametre_objet,
                                self.x_target + self.diametre_objet,
                                self.y_target + self.diametre_objet,
                                fill='red'
                                )

        # Création d'un menu pour générer une nouvelle cible
        menuBar = Menu(self.canvas)
        menuFichier = Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label="Demarrer", menu=menuFichier)
        menuFichier.add_command(label="Nouvelle cible",
                                command=self.set_target)
        self.tk.config(menu=menuBar)

        # Génération de la première cible
        self.set_target(550, 600)

        self.list_boutton = list()
        self.list_boutton.append(self.Boutton(self.canvas, 100, 75, 'M1-'))
        self.list_boutton.append(self.Boutton(self.canvas, 170, 75, 'M1+'))

        self.list_boutton.append(self.Boutton(self.canvas, 300, 75, 'M2-'))
        self.list_boutton.append(self.Boutton(self.canvas, 370, 75, 'M2+'))

        self.list_boutton.append(self.Boutton(self.canvas, 500, 75, 'M3-'))
        self.list_boutton.append(self.Boutton(self.canvas, 570, 75, 'M3+'))



    def set_target(self, x=None, y=None):
        """
        Fonction qui génère une nouvelle cible et la positionne sur le canvas.

        Args:
            x (int, optional): Position horizontale de la cible sur le canvas. Si non spécifié, une position aléatoire est générée.
            y (int, optional): Position verticale de la cible sur le canvas. Si non spécifié, une position aléatoire est générée.

        Raises:
            UserWarning: Si la position de la cible est hors du canvas.

        Returns:
            None

        """

        if 'x' in locals() and 'y' in locals() and 0 < x < self.largeur and 0 < y < self.hauteur:
            # Vérification si les arguments x et y sont spécifiés et sont valides
            # Génération de nouvelles coordonnées pour la cible
            self.x_target = x
            self.y_target = y
            print(f"Cible générée à la position {x}, {y}.")

        else:
            # Sinon, générer une position aléatoire pour la cible
            if 0 < x < self.largeur or 0 < y < self.hauteur:
                warnings.warn("Cible hors Canvas.")
            self.x_target = random.randrange(0, self.largeur)
            self.y_target = random.randrange(0, self.hauteur)
            print("Cible générée aléatoirement.")

        # Suppression de l'ancienne cible et création d'une nouvelle
        self.canvas.delete(self.target)
        self.target = self.canvas.create_oval(self.x_target - self.diametre_objet,
                                              self.y_target - self.diametre_objet,
                                              self.x_target + self.diametre_objet,
                                              self.y_target + self.diametre_objet,
                                              fill='red'
                                              )





    class Boutton():
        def __init__(self, canvas, x_position_centre, y_position_centre, text=""):
            """
            Initialise un objet Boutton avec des informations sur sa position et son aspect graphique.

            :param canvas: objet Canvas sur lequel le bouton sera dessiné
            :param x_position_centre: position horizontale du centre du bouton
            :param y_position_centre: position verticale du centre du bouton
            :param text: identifiant du bouton (texte affiché dans le bouton)
            """
            self.canvas = canvas

            # Définition des coordonnées du rectangle englobant le bouton
            self.taille_boutton = 25
            self.x1 = x_position_centre - self.taille_boutton
            self.y1 = y_position_centre - self.taille_boutton
            self.x2 = x_position_centre + self.taille_boutton
            self.y2 = y_position_centre + self.taille_boutton

            # Création du rectangle et du texte du bouton sur le canvas
            self.rectangle = self.canvas.create_rectangle(self.x1, self.y1,
                                                          self.x2, self.y2,
                                                          fill='ivory')
            self.id = text
            self.name = self.canvas.create_text(x_position_centre,
                                                y_position_centre,
                                                anchor="center",
                                                text=text,
                                                font=("Arial", 15))

        def est_dedans(self, x_target, y_target):
            """
            Méthode pour vérifier si un point est situé à l'intérieur du bouton.

            Args:
                x_target (int): La coordonnée x du point cible.
                y_target (int): La coordonnée y du point cible.

            Returns:
                int: 1 si le point est à l'intérieur du bouton, 0 sinon.
            """
            # Vérifie si les coordonnées x et y du point cible sont à l'intérieur de la zone du bouton.
            if self.x1 < x_target < self.x1 + 2 * self.taille_boutton and self.y1 < y_target < self.y1 + 2 * self.taille_boutton:
                return 1
            # Si le point n'est pas à l'intérieur du bouton, renvoie 0.
            return 0

if __name__ == '__main__':
    tk = Tk()
    Environnement(tk)