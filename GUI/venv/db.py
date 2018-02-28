import sqlite3 as sql

def insert_search(uname, numOfTweets):
    con = sql.connect("database.db")
    cur = con.cursor()
    cur.execute("INSERT INTO search (uname, numOfTweets) VALUES (?, ?)", (uname, numOfTweets))
    print
    con.commit()
    con.close()

def insert_tweets(search_id, tweet_info):



def select_search():
    con = sql.connect("database.db")
    cur = con.cursor()

    result = cur.execute("SELECT * FROM search")

    print(result.fetchall())

    con.close()

def main():
    insert_search("Kalle Kula", 10)
    select_search()





if __name__ == "__main__":
    main()