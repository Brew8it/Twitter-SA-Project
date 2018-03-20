from TSA.Preproc import Preproc
from TSA.NB import NB
from TSA.SVM import SVM
from TSA.Prediction import Prediction
from TSA.GUI import app
from TSA.CNN import CNN
from TSA.TwitterMiner import TwitterMiner

import sys, os

# Preproc.main()
# NB.train_NB()
# SVM.train_SVM()
# Prediction.main()
# app.main()
# app.main()
# CNN.train_CNN()
# TwitterMiner.main()


def main_menu():
    print("Welcome to Brew8it and Selberget TSA program")
    print("Please choose the menu you want to start:")
    print("1 Train Naive Bayes")
    print("2 Train SVM")
    print("3 Train CNN")
    print("4 start webserver for GUI")
    print("5 Test Pred")
    print("0 Exit")
    choise = input()
    exec_menu(choise)

def exec_menu(choise):
    if choise == "1":
        NB.train_NB()
    elif choise == "2":
        SVM.train_SVM()
    elif choise == "3":
        CNN.train_CNN()
    elif choise == "4":
        gui()
    elif choise == "5":
        Prediction.main()

def gui():
    app.main()



if __name__ == '__main__':
    main_menu()
