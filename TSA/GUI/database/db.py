import sqlite3 as sql


def insert_search_with_tweets(uname, numOfTweets, predictedTweets):
    with sql.connect("TSA/GUI/database/database.db") as con:
        cur = con.cursor()
        # Insert search record
        cur.execute("INSERT INTO search (uname, numOfTweets) VALUES (?, ?)", (uname, numOfTweets))
        search_id = cur.lastrowid
        # print(predictedTweets)
        # Insert tweet record
        for predictedTweet in predictedTweets:
            cur.execute(
                "INSERT INTO tweets (tweet, NBSE, NBSTS, SVMSE, SVMSTS, CNNSE, CNNSTS, AVG, search_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (predictedTweet[0], predictedTweet[1], predictedTweet[2], predictedTweet[3], predictedTweet[4],
                 predictedTweet[5],
                 predictedTweet[6], predictedTweet[7], search_id))
        con.commit()


def get_search_records():
    with sql.connect("TSA/GUI/database/database.db") as con:
        cur = con.cursor()
        result = cur.execute("SELECT * FROM search")
        return result.fetchall()


def get_tweets_records():
    with sql.connect("TSA/GUI/database/database.db") as con:
        cur = con.cursor()
        result = cur.execute("SELECT tweet, NBSE, NBSTS, SVMSE, SVMSTS, CNNSE, CNNSTS, AVG FROM tweets")
        return result.fetchall()


def get_posneg():
    classifierSentiment = {}
    classifiers = ["NBSE", "NBSTS", "SVMSE", "SVMSTS", "CNNSE", "CNNSTS"]

    for classifier in classifiers:
        classifierSentiment[classifier] = get_pos_neg_count(classifier)

    return classifierSentiment


def get_pos_neg_count(classifier):
    with sql.connect("TSA/GUI/database/database.db") as con:
        cur = con.cursor()
        lst = []
        lst.append(cur.execute("SELECT COUNT(*) FROM tweets WHERE " + classifier + " = 1").fetchone()[0])
        lst.append(cur.execute("SELECT COUNT(*) FROM tweets WHERE " + classifier + " = 0").fetchone()[0])
        return lst


def clear_searches():
    with sql.connect("TSA/GUI/database/database.db") as con:
        con.execute("PRAGMA foreign_keys = ON")  # Has to be enabled manually for every connection
        cur = con.cursor()
        cur.execute("DELETE FROM search")
        # cur.execute("DELETE FROM tweets")


def main():
    clear_searches()
