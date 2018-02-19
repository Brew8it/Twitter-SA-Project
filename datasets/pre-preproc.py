import pandas as pd


data = pd.read_csv("../datasets/SemEval/4A-English/SemEval2017-task4-dev.subtask-A.english.INPUT.txt", sep="\t", header=None, names=["a", "b", "c"])

print data['b'] +" " + data['c']