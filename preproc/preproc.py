# coding=utf-8
import os
import re
import pandas as pd
from miner import miner
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer

import unicodedata

# python -m spacy download en
# nlp = spacy.load('en', parse=False, tag=False, entity=False)

pd.options.display.max_colwidth = 200

stemmer = SnowballStemmer('english')
stop = stopwords.words('english')
stop.remove('not')

# for expanding contractions

cont_dict = {'can\'t': 'cannot', 'won\'t': 'will not', 'n\'t': ' not'}
cont_re = re.compile('(%s)' % '|'.join(cont_dict.keys()))


class preProc(object):
    def __init__(self):
        self.df = pd.DataFrame

    def loadCsv(self, path, name):
        os.chdir(path)
        self.df = pd.read_csv(name, sep=",").dropna()
        self.df.columns = ["num", "lable", "tweet"]

    def loadOwnDataFrame(self, dataframe):
        self.df = dataframe.dropna()

    def expand_cont(self, s, cont_dict=cont_dict):
        def replace(match):
            return cont_dict[match.group(0)]

        return cont_re.sub(replace, s)

    def clean_data(self, html_strpping=True, accented_char_removal=True, to_lower=True, remove_links=True,
                   remove_mentions=True,
                   remove_hashtag=True, remove_extra_whitespace=True, tokenize=True, stemming=True,
                   remake_document=True,
                   expand_contractions=True):
        if html_strpping:
            self.remove_html_encode()
        if to_lower:
            self.to_lower()
        if expand_contractions:
            self.expand_contractions()
        if accented_char_removal:
            self.remove_accented_chars()
        if remove_links:
            self.remove_links()
        if remove_mentions:
            self.remove_twitter_mention()
        if remove_hashtag:
            self.remove_hashtag()
        if remove_extra_whitespace:
            self.remove_extra_whitepsace()
        if tokenize:
            self.tokenize()
        if stemming:
            self.word_stemming()
        if stopwords:
            self.remove_stopwords()
        if remake_document:
            self.remake_tweets()

    def get_twitter_df(self):
        # return self.df[['lable', "tweet"]]
        return self.df

    def remove_html_encode(self):
        # https://stackoverflow.com/questions/44703945/pandas-trouble-stripping-html-tags-from-dataframe-column
        self.df["tweet"] = self.df["tweet"].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

    def remove_accented_chars(self):
        self.df["tweet"] = self.df["tweet"].apply(
            lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))

    def to_lower(self):
        self.df["tweet"] = self.df["tweet"].str.lower()

    def remove_links(self):
        self.df.loc[:, "tweet"] = self.df.loc[:, "tweet"].replace(r'https?://[A-Za-z0-9./]+', '', regex=True)

    def remove_twitter_mention(self):
        self.df.loc[:, "tweet"] = self.df.loc[:, "tweet"].replace(r'@[A-Za-z0-9]+', '', regex=True)

    def remove_hashtag(self):
        self.df.loc[:, "tweet"] = self.df.loc[:, "tweet"].replace('[^a-zA-Z]', ' ', regex=True)

    def remove_extra_whitepsace(self):
        self.df.loc[:, "tweet"] = self.df.loc[:, "tweet"].replace(' +', ' ', regex=True)

    def expand_contractions(self):
        self.df["tweet"] = self.df["tweet"].apply(lambda x: self.expand_cont(x))

    def tokenize(self):
        # self.df["tweet"] = self.df.apply(lambda row: word_tokenize(row["tweet"]), axis=1) bajs lambda
        self.df["tweet"] = self.df["tweet"].apply(word_tokenize)

    def word_stemming(self):
        self.df["tweet"] = self.df["tweet"].apply(lambda x: [stemmer.stem(y) for y in x])

    def remove_stopwords(self):
        self.df["tweet"] = self.df["tweet"].apply(lambda x: [item for item in x if item not in stop])

    def remake_tweets(self):
        self.df["tweet"] = self.df["tweet"].apply(lambda x: ' '.join(x))


# for testing

if __name__ == "__main__":
    test = preProc()
    # test.loadCsv("../datasets/STS/", "STS_test.csv")
    # test.loadCsv("../datasets/SemEval/4A-English/", "SemEval.csv")

    df = pd.DataFrame(
        np.array([[1, "you don\'t need didn\'t"], [2, "This is a test tweet with you don\'t need didn\'t"]]))
    df.columns = ["lable", "tweet"]
    test.loadOwnDataFrame(df)
    test.clean_data()
    df = test.get_twitter_df()
    print(df)

    """
    test = preProc()
    tweet_miner = miner.TwitterMiner()
    # tweet_miner.collect_tweets_from_searching("#svpol", 10)
    df = tweet_miner.collect_tweets_from_user_feed("danlevene", 10)
    test.loadOwnDataFrame(df[["date", "length", "tweet"]])
    test.clean_data()
    df = test.get_twitter_df()
    print(df)
    """
