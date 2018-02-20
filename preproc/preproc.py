import os

import pandas as pd

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
stop = stopwords.words('english')


class preProc(object):
    def __init__(self):
        self.df = pd.DataFrame

    def loadCsv(self, path, name):
        os.chdir(path)
        self.df = pd.read_csv(name, header=None).dropna()
        self.df.columns = ["num","lable", "tweet"]

    def clean_data(self):
        self.remove_links()
        # test remove html encoding
        self.remove_html_encode()
        # test remove twitter mention e.g @username
        self.remove_twitter_mention()
        # test remove hashtags, numbers, and special chars
        self.remove_hashtag()
        # test for tokenize
        self.tokenize()
        # test word stemming
        self.word_stemming()
        self.remove_stopwords()

    def get_twitter_df(self):
        return self.df[['lable', "tweet"]]

    def clean_twitter_data(self):
        # call the bellow functions
        return self.df

    def remove_links(self):
        self.df.loc[:, "tweet"] = self.df.loc[:, "tweet"].replace(r'https?://[A-Za-z0-9./]+', '', regex=True)

    def remove_html_encode(self):
        # https://stackoverflow.com/questions/44703945/pandas-trouble-stripping-html-tags-from-dataframe-column
        self.df["tweet"] = self.df["tweet"].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

    def remove_twitter_mention(self):
        self.df.loc[:, "tweet"] = self.df.loc[:, "tweet"].replace(r'@[A-Za-z0-9]+', '', regex=True)

    def remove_hashtag(self):
        self.df.loc[:, "tweet"] = self.df.loc[:, "tweet"].replace('[^a-zA-Z]', ' ', regex=True)

    def tokenize(self):
        # https://stackoverflow.com/questions/33098040/how-to-use-word-tokenize-in-data-frame
        # self.df["tweet"] = self.df.apply(lambda row: word_tokenize(row["tweet"]), axis=1)
        self.df["tweet"] = self.df["tweet"].str.lower().str.split()

    def word_stemming(self):
        self.df["tweet"] = self.df["tweet"].apply(lambda x: [stemmer.stem(y) for y in x])


    def remove_stopwords(self):
        self.df["tweet"] = self.df["tweet"].apply(lambda x: [item for item in x if item not in stop])


# for testing

if __name__ == "__main__":
    test = preProc()
    test.loadCsv("../datasets/SemEval/4A-English/", "SemEval.csv")
    test.clean_data()
    df = test.get_twitter_df()
    print df
