from neural_network import Neural_network
from tkinter import *
from environnement import Environnement
from bras_robot import BrasRobot
import matplotlib.pyplot as plt
# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    tk = Tk()
    env = Environnement(tk)

    robot = BrasRobot(env, 50, 750)
    q_neuronne = Neural_network(robot)
    # #
    q_neuronne.main_train_bras_robot(cycle=100, step=200)
    # q_neuronne.
    plt.plot(q_neuronne.result_cycle)
    plt.show()

    # q_neuronne.test(name_model="last_model.h5")


