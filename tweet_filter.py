"""Process the text of the tweets using FROG."""
import sqlite3
import requests
import time
import re
import json
from tqdm import tqdm

from resources import config

DB_FILE = r"resources/tweets.db"

def process_tweets():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM "tweets"')
    tweets = c.fetchall()

    # remove retweets
    tweets = [tweet for tweet in tweets if not tweet[2].startswith("RT")]

    # remove all non-dutch tweets
    tweets = [tweet for tweet in tweets if tweet[3] == "nl"]
    
    # generate tokens, remove links and usernames
    token_tweets = []
    for tweet in tweets:
        tokens = tweet[2].split()
        tokens = [re.sub('@(\w){1,15}', 'USERNAME', token) for token in tokens]
        tokens = [re.sub('(http:)?//t.co/\w*', 'LINK', token) for token in tokens]
        token_tweets.append([tweet[0], tokens, tweet[1], tweet[4]])
    tweets = token_tweets

    # remove truncated tweets that end with link
    tweets = [tweet for tweet in tweets if not tweet[1][-1].endswith("LINK")]
   
    # only allow 1 tweet per user per day
    days = {}
    daily_tweets = []
    for tweet in tweets:
        day = tweet[3]
        if day in days:
            if tweet[2] not in days[day]:
                days[day].add(tweet[2])
                daily_tweets.append(tweet)             
        else:
            days[day] = set(tweet[2])
            daily_tweets.append(tweet)
    tweets = daily_tweets

    # store tweets in db
    c.executemany(
        'INSERT INTO "labels" (id, text) VALUES (?,?)',
        ([(tweet[0], json.dumps(tweet[1])) for tweet in tweets]))

    conn.commit()
    conn.close()

    print("Done")


def get_RT_ids():
    """Retrieve retweet ids."""
    # This method does not seem to work correctly. The retweet ids are different from what they should be.
    print("Retreiving tweets...")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM "tweets"')
    tweets = c.fetchall()

    tweets = [tweet[0] for tweet in tweets if tweet[2].startswith("RT")]
    bulk_tweets = [tweets[i:i+100] for i in range(0, len(tweets), 100)]

    print(f"retrieving {len(tweets)} tweets")

    adress = 'https://api.twitter.com/1.1/statuses/lookup.json'
    headers = {"Authorization": f"Bearer {config.BEARER_TOKEN}"}
    result = []
    failed = []    

    for group in tqdm(bulk_tweets):
        
        tweet_ids = [tweet for tweet in group]
        params = "?id=" + ','.join(map(str, tweet_ids))
        r = requests.get(adress+params, headers=headers)

        while r.status_code == 429:
            cooldown = 5
            print(f"Too many requests rechecking in {cooldown} minute(s)")
            for i in tqdm(range(60 * cooldown)):
                time.sleep(1)

            r = requests.get(adress+params, headers=headers)

        if r.status_code == 200:
            rts = r.json()

            status_ids = []
            for tweet_id, rt in zip(tweet_ids, rts):
                # skip retweets ids that are not present
                try:
                    status_ids.append((tweet_id, rt["retweeted_status"]["id_str"]))
                except KeyError as e:
                    failed.append(tweet_id)
                    continue
            result.extend(status_ids)
        else:
            print(r)

    print("Inserting into the database")
    c.executemany(
        'UPDATE tweets SET rt_id = ? WHERE id = ?',
        ([(r[1], r[0]) for r in result]))

    conn.commit()
    conn.close()

    f = open(r"tweet.txt", 'wt')
    f.write('\n'.join(failed))
    f.close()

    print("Done")
