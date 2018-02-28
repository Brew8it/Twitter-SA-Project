import sqlite3 as sql


def insert_search_with_tweets(uname, numOfTweets, predictedTweets):
    with sql.connect("database/database.db") as con:
        cur = con.cursor()
        # Insert search record
        cur.execute("INSERT INTO search (uname, numOfTweets) VALUES (?, ?)", (uname, numOfTweets))
        # Insert tweet record
        print(predictedTweets)

        for predictedTweet in predictedTweets:
            print(predictedTweet)
            cur.execute("INSERT INTO tweets (tweet, NBSE, NBSTS, SVMSE, SVMSTS, search_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (predictedTweet[0], predictedTweet[1], predictedTweet[2], predictedTweet[3], predictedTweet[4],
                     cur.lastrowid))
        con.commit()


def get_search_records():
    with sql.connect("database/database.db") as con:
        cur = con.cursor()
        result = cur.execute("SELECT * FROM search")
        return result.fetchall()


def get_tweets_records():
    with sql.connect("database/database.db") as con:
        cur = con.cursor()
        result = cur.execute("SELECT * FROM tweets")
        return result.fetchall()


def get_posneg():
    classifierSentiment = {}
    classifiers = ["NBSE", "NBSTS", "SVMSE", "SVMSTS"]

    for classifier in classifiers:
        classifierSentiment[classifier] = get_pos_neg_count(classifier)

    return classifierSentiment


def get_pos_neg_count(classifier):
    with sql.connect("database/database.db") as con:
        cur = con.cursor()
        lst = []
        lst.append(cur.execute("SELECT COUNT(*) FROM tweets WHERE " + classifier + " = 1").fetchone()[0])
        lst.append(cur.execute("SELECT COUNT(*) FROM tweets WHERE " + classifier + " = 0").fetchone()[0])
        return lst

def clear_searches():
    with sql.connect("database/database.db") as con:
        con.execute("PRAGMA foreign_keys = ON") # Has to be enabled manually for every connection
        cur = con.cursor()
        cur.execute("DELETE FROM search")

def main():
    clear_searches()
    #insert_search_with_tweets("Kalle Kula", 10, 2)
    print("Search")
    print(get_search_records())
    print("Tweets")
    print(get_tweets_records())
    print("Negative tweets:")
    get_posneg()

    #insert_search_with_tweets("asdf", 12, [["asdasd", 1, 0, 1, 1]])

if __name__ == "__main__":
    main()
