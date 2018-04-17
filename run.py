from TSA.Preproc import Preproc
from TSA.NB import NB
from TSA.SVM import SVM
from TSA.Prediction import Prediction
from TSA.GUI import app
from TSA.CNN import CNN
from TSA.TwitterMiner import TwitterMiner
from TSA.NB import NB_improved_SE
from TSA.NB import NB_improved_STS
from TSA.SVM import SVM_improved_SE, SVM_improved_STS
from TSA.Lexicon import Lexicon
import sys, os




def main_menu():
    print("Welcome to Brew8it and Selberget TSA program")
    print("Please choose the menu you want to start:")
    print("1 Train Naive Bayes")
    print("2 Train SVM")
    print("3 Train CNN")
    print("4 start webserver for GUI")
    print("5 Test Pred")
    print("6 Save preproc to csv")
    print("7 Train improved NB_SE")
    print("8 Train improved NB_STS")
    print("9 Train improved SVM_SE")
    print("10 Train improved SVM_STS")
    print("11 Run Lexicon")
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
    elif choise == "6":
        Preproc.main()
    elif choise == "7":
        NB_improved_SE.train_NB()
    elif choise == "8":
        NB_improved_STS.train_NB()
    elif choise == "9":
        SVM_improved_SE.train_SVM()
    elif choise == "10":
        SVM_improved_STS.train_SVM()
    elif choise == "11":
        Lexicon.main()


def gui():
    app.main()


if __name__ == '__main__':
    main_menu()
