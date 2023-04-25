


import random

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



class Action_moteur():

    """
    Cette classe représente une liste d'action possible à effectuer sur un objet.

    Chaque action possible est une combinaison de trois valeurs (i, j, k) où i, j, et k sont compris entre -1 et 1.
    Les valeurs de i, j et k représentent respectivement la direction de rotation de 3 moteurs.
    Les actions possibles sont stockées dans une liste accessible via l'attribut 'action_possible' de l'objet.

    Cette classe propose également une fonction pour choisir une action au hasard parmi les actions possibles.
    """

    def __init__(self):
        """
        Initialise l'objet Action avec une liste des actions possibles.

        """

        # Crée une liste vide pour stocker les actions possibles
        self.action_possible = list()

        self.action_possible = [
            (-1, 0, 0),
            (1, 0, 0),
            # (-1, -1, 0),
            (0, -1, 0),
            (0, 1, 0),
            # (1, 1, 0),
            (0, 0, -1),
            # (0, -1, -1),
            # (-1, -1, -1),
            (0, 0, 1),
            # (0, 1, 1),
            # (1, 1, 1),
        ]



    def get_random_index(self):
        """
        Cette fonction renvoie un index aléatoire compris entre 0 et la longueur de la liste d'actions possibles (self.action_possible).

        Args:
            Aucun argument requis.

        Returns:
            Un entier représentant l'index aléatoire.
        """
        # Générer un nombre entier aléatoire compris entre 0 et la longueur de la liste d'actions possibles.
        indice = random.randrange(0, len(self.action_possible))
        # Renvoyer l'indice aléatoire généré.
        return indice

    def get_action(self, indice=None) -> int:
        """
        Cette fonction renvoie une action possible dans la liste des actions possibles de l'objet.

        Arguments:
        - indice : un entier optionnel représentant l'indice de l'action souhaitée dans la liste des actions possibles.
          Si l'indice est None (par défaut), une action aléatoire est choisie.
          Si l'indice est fourni, la fonction vérifie s'il est valide (i.e. s'il est dans les limites de la liste des actions possibles),
          et renvoie l'action correspondante. Si l'indice n'est pas valide, la fonction émet un avertissement et renvoie une action aléatoire.

        Retour:
        - L'action choisie (un tuple de taille 3).
        """

        if indice is None:
            # Si l'indice n'est pas fourni, choisir une action aléatoire
            indice = random.randrange(0, len(self.action_possible))
        else:
            # Si l'indice est fourni, vérifier s'il est valide
            if indice >= len(self.action_possible) or indice < 0:
                # Si l'indice n'est pas valide, émettre un avertissement et choisir une action aléatoire
                raise ArithmeticError(f"impossible d'effectuer l'action d'indice {indice}")

        # Renvoyer l'action correspondante
        return self.action_possible[indice]

    def get_mask_action(self, df_action: pd.DataFrame):
        """
        Crée un masque booléen pour spécifier quelle action a été prise.

        Args:
            df_action: Un dataframe de dimension n x 1, où chaque ligne représente l'indice de l'action prise.
                Les indices doivent être des entiers entre 0 et la longueur de la liste `action_possible`.

        Returns:
            Un dataframe de dimension n x len(self.action_possible), où chaque ligne représente une action prise.
            Le dataframe a une valeur de 1 pour l'indice d'action spécifié et des zéros pour les autres actions.

        Raises:
            AttributeError: Si `df_action` n'est pas un dataframe de dimension n x 1,
                ou si un indice d'action spécifié est invalide (en dehors de l'intervalle [0, len(self.action_possible)]).
        """

        # Vérifier la forme du dataframe
        nbr_ligne, nbr_colonne = df_action.shape
        if nbr_colonne != 1 or nbr_ligne is None:
            raise AttributeError(
                f"df_action doit etre un dataframe de dim n x 1 [nombre de colonne: {nbr_colonne}]")

        mask_action_possible = pd.DataFrame()

        for i in range(len(df_action)):

            index_action = df_action.iloc[i, 0]

            # Vérifier que l'indice d'action spécifié est valide
            if index_action > len(self.action_possible) or index_action < 0:
                raise AttributeError(f'Specifier un index valide [{index_action}].')

            # Créer un masque numpy qui a une valeur de 1 pour l'index d'action spécifié, et des zéros pour les autres actions
            nouveau_df_ligne = np.zeros(shape=(1, len(self.action_possible)))
            nouveau_df_ligne[0, index_action] = 1
            nouveau_df_ligne = pd.DataFrame(nouveau_df_ligne)

            mask_action_possible = pd.concat((mask_action_possible, nouveau_df_ligne), axis=0)

        if nbr_ligne != len(mask_action_possible):
            raise ArithmeticError(f"Incoheence de la matrice mask cree {nbr_ligne}, {len(mask_action_possible)}")

        return mask_action_possible


if __name__ == '__main__':

    act = Action_moteur()