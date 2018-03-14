from TSA.TwitterMiner import keys
import tweepy
import pandas as pd
import numpy as np



# from IPython.display import display

pd.options.display.max_colwidth = 300


class TwitterMiner:
    """Class for mining tweets from twitter"""

    def __init__(self):
        self.api = self.setup()

    def setup(self):
        """Function to setup Twitter API with the keys provided"""
        auth = tweepy.OAuthHandler(keys.CONSUMER_KEY, keys.CONSUMER_SECRET)
        auth.set_access_token(keys.ACCESS_TOKEN, keys.ACCESS_SECRET)

        return tweepy.API(auth, wait_on_rate_limit=True)

    def collect_tweets_from_searching(self, search_string, number_of_tweets):
        tweets = tweepy.Cursor(self.api.search, q=search_string, lang="en").items(number_of_tweets)
        df = self.create_data_frame(tweets)
        return df

    def collect_tweets_from_user_feed(self, username, number_of_tweets):
        # Use the tweet_mode='extended' to be able to get full length tweets.
        # Twitter has made API changes that tweets will be truncated after 140 chars is tweet_mode is not specified
        tweets = self.api.user_timeline(screen_name=username, count=number_of_tweets, tweet_mode='extended')
        df = self.create_data_frame(tweets)
        return df
        # df.to_csv("tweets.csv")

    def create_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=['tweet'])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['length'] = np.array([len(tweet.full_text) for tweet in tweets])

        return df


# Test

def main():
    tweet_miner = TwitterMiner()
    # tweet_miner.collect_tweets_from_searching("#svpol", 10)
    twitterdf = tweet_miner.collect_tweets_from_user_feed("realdonaldtrump", 1)
    print(twitterdf)

