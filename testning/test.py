import re

import pandas as pd
import numpy as np

cont_dict = {'can\'t': 'cannot', 'won\'t': 'will not', 'n\'t': ' not'}
cont_re = re.compile('(%s)' % '|'.join(cont_dict.keys()))


def expand_cont(s, cont_dict=cont_dict):
    def replace(match):
        return cont_dict[match.group(0)]

    return cont_re.sub(replace, s)


# print(expand_cont('you don\'t need didn\'t'))

df = pd.DataFrame(np.array([[1, "you don\'t need didn\'t :: can\'t, won\'t"], [2, "This is a test tweet with you don\'t need didn\'t"]]))

df[1] = df[1].apply(lambda x: expand_cont(x))

print(df)



"""

self.df["tweet"] = self.df["tweet"].apply(lambda x: [item for item in x if item not in stop])

self.df["tweet"] = self.df["tweet"].apply(lambda x: [stemmer.stem(y) for y in x])

if x 

"""
