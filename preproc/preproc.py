import os

import pandas as pd

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk import word_tokenize

pd.options.display.max_colwidth = 190


class preProc(object):
    def __init__(self):
        self.df = pd.DataFrame

    def loadCsv(self, path, name):
        os.chdir(path)
        self.df = pd.read_csv(name).dropna()

    def get_twitter_df(self):
        return self.df

    def clean_twitter_data(self):
        # call the bellow functions
        return self.df

    def remove_links(self):
        self.df.loc[:, "c"] = self.df.loc[:, "c"].replace(r'https?://[A-Za-z0-9./]+', '', regex=True)

    def remove_html_encode(self):
        # https://stackoverflow.com/questions/44703945/pandas-trouble-stripping-html-tags-from-dataframe-column
        self.df['c'] = self.df['c'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

    def remove_twitter_mention(self):
        self.df.loc[:, "c"] = self.df.loc[:, "c"].replace(r'@[A-Za-z0-9]+', '', regex=True)

    def remove_hashtag(self):
        self.df.loc[:, "c"] = self.df.loc[:, "c"].replace('[^a-zA-Z]', ' ', regex=True)

    def tokenize(self):
        for tweet_sw in temp_df.loc[:, "text"]:
            tweet_sw_tokenized = nltk.word_tokenize(tweet_sw)


    def remove_stopwords(self):
        # use nltk
        return self.df


# for testing


test = preProc()
test.loadCsv("../datasets/SemEval/4A-English/", "SemEval.csv")
# test remove links
test.remove_links()
# test remove html encoding
test.remove_html_encode()

# test remove twitter mention e.g @username
test.remove_twitter_mention()

# test remove hashtags, numbers, and special chars
test.remove_hashtag()

df = test.get_twitter_df()

print df.c[:50]

# satt headers i read_csv
