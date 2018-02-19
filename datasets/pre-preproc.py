import pandas as pd

import numpy as np
import os

os.chdir('../datasets/SemEval/4A-English/')

data = pd.read_csv("SemEval2017-task4-dev.subtask-A.english.INPUT.txt", sep="\t", header=None, names=["a", "b", "c"], keep_default_na=False, na_values=[';'])

data = data[data.b != "neutral"]

data['lable'] = np.where(data['b'] == 'positive', '1', '0')

#print data['c']

headers = ["lable", 'c']
#data.to_csv('SemEval.csv', columns=headers)


os.chdir('../../STS/')



df = pd.read_csv("training.1600000.processed.noemoticon.csv", sep=",", names=["l","id", "date", "q", "username", "tweet"])

df["lable"] = np.where(df["l"] == 4, "1", "0")

headers = ["lable", "tweet"]



df.to_csv('STS.csv', columns=headers)
