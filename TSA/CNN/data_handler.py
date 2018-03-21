import itertools
from collections import Counter
import numpy as np
from TSA.Preproc import Preproc


def split_tweets_by_words(tweets):
    [tweet.split(" ") for tweet in tweets]
    return tweets


def get_max_tweet_sequence(tweets):
    return max(len(tweet) for tweet in tweets)


def get_tweets_as_numbers(tweets, vocabulary):
    return np.array([[vocabulary[word] for word in tweet] for tweet in tweets])


def pad_tweet(tweet, max_length, padding="<PAD/>"):
    return tweet + [padding] * (max_length - len(tweet))


def create_vocabulary(tweets):
    counts_of_word = Counter(itertools.chain(*tweets))

    inverse_vocabulary = [x[0] for x in counts_of_word.most_common()]
    inverse_vocabulary = list(sorted(inverse_vocabulary))

    vocabulary = {x: i for i, x in enumerate(inverse_vocabulary)}

    return [vocabulary, inverse_vocabulary]


def pred_load_data(df):
    # Convert tweet to list of words and apply padding
    df["tweet"] = df["tweet"].apply(lambda x: x.split(" "))
    # set max length to a fixed size..
    # max_length = get_max_tweet_sequence(df.tweet)
    max_length = 64
    df["tweet"] = df["tweet"].apply(lambda x: pad_tweet(x, max_length))
    vocabulary, inverse_vocabulary = create_vocabulary(df.tweet)
    x = get_tweets_as_numbers(df.tweet, vocabulary)
    return x




def load_data(df):
    # Convert tweet to list of words and apply padding
    df["tweet"] = df["tweet"].apply(lambda x: x.split(" "))
    # max_length = get_max_tweet_sequence(df.tweet)
    max_length = 64
    df["tweet"] = df["tweet"].apply(lambda x: pad_tweet(x, max_length))

    df['lable'] = df['lable'].apply(lambda d: [1, 0] if d == 1 else [0, 1])

    y = df['lable'].tolist()
    y = np.array(y)

    vocabulary, inverse_vocabulary = create_vocabulary(df.tweet)

    x = get_tweets_as_numbers(df.tweet, vocabulary)

    # return [x, df.lable, vocabulary, inverse_vocabulary]
    return [x, y, vocabulary, inverse_vocabulary]
