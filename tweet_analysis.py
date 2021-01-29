"""Download and analyse tweets."""
import sqlite3
import argparse
import time
import requests
import json

from datetime import datetime
from datetime import timezone

import numpy as np
import classifier
from tqdm import tqdm
import tweet_filter
from resources import config

#17862 in first batch
DB_FILE = r"resources\tweets.db"
LOG = r"resources\log.txt"
TERMS = [
    "RIVM",
    "persconferentie",
    "coronamaatregelen",
    "persco"
]


def setup():
    """Create the database and table."""
    print("Creating database...")
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE "tweets" (
        "id"        text,
        "author_id" text,
        "text"      text,
        "language"  text,
        "created_at" DATETIME,
        "tag"   text,
        PRIMARY KEY("id")
            )''')
    conn.execute('''CREATE TABLE "frog_NLTK" (
        "tweet_id"  text,
        "output"    text,
        PRIMARY KEY("tweet_id")
            )''')
    conn.execute('''CREATE TABLE "labels" (
        "id"        text,
        "text"      text,
        "label"     integer,
        "annotated  integer"
        PRIMARY KEY("id")
            )''')


    conn.commit()
    conn.close()


def get_tweets(params):
    """
    TODO.

    :raises NotImplementedError: [description]
    """
    adress = 'https://api.twitter.com/1.1/search/tweets.json'
    headers = {"Authorization": f"Bearer {config.BEARER_TOKEN}"}

    # f = open('resources/Example.json')
    # tweets = json.load(f)
    # f.close()

    r = requests.get(adress+params, headers=headers)

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"[{time}] GET: {adress+params} {r.status_code}")

    try:
        tweets = r.json()
        store_tweets(tweets["statuses"], tweets["search_metadata"]["query"][3:])  # Removes %23 before tag

        if r.status_code == 200:
            params = tweets["search_metadata"]["next_results"]

    except KeyError as e:
        log(f'retreived all terms for: {tweets["search_metadata"]["query"][3:]}')
        return None

    return params


def log(statement):
    f = open(LOG, 'a')
    print(statement)
    f.write(statement + '\n')
    f.close()


def store_tweets(tweets, tag):
    """
    Store tweets in the database.

    :param tweets: Tweets from twitter API
    :type tweets: Array of tweets
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.executemany(
        'INSERT or IGNORE INTO tweets (id, author_id, text, language, tag, created_at) VALUES (?,?,?,?,?,?)',
        ([(tweet["id_str"], tweet["user"]["id_str"], tweet["text"], tweet["lang"], tag,
            datetime.strptime(tweet["created_at"], r"%a %b %d %H:%M:%S %z %Y")) for tweet in tweets]))
    conn.commit()
    conn.close()
    log(f'{len(tweets)} Tweets stored (for {tag})')


def loop_tweets():
    since_id = find_since()

    print(f"starting from: {since_id}")

    if len(since_id) != len(TERMS):
        raise NotImplementedError("no implementation for missing tags")

    params = [f'?q=%23{q}&count=100&result_type=recent&since_id={since_id[q]}' for q in TERMS]

    running = [True] * len(TERMS)

    while any(running):
        i = 0
        for i in range(len(TERMS)):
            if not running[i]:
                continue
            params[i] = get_tweets(params[i])
            if params[i] is None:   # To run unneeded code
                running[i] = False
                continue

            params[i] += f'&since_id=' + since_id[TERMS[i]]
        print("\n")
        time.sleep(1)

    print("done")


def find_since():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT tag, MAX(id) FROM tweets GROUP BY tag')
    r = c.fetchall()
    conn.close()
    return {key: value for key, value in r}


def tweet_analysis():
    """
    TODO.

    :raises NotImplementedError: [description]
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM "tweets" INNER JOIN "labels" ON tweets.id = labels.id INNER JOIN "frog_NLTK" ON tweets.id = frog_NLTK.tweet_id')
    
    tweets_raw = c.fetchall()

    conn.close()

    dates = [datetime(2020, 11, 17, tzinfo=timezone.utc), datetime(2020, 12, 8, tzinfo=timezone.utc)]

    groups = [[], [], []]
    for t_raw in tweets_raw:
        t = {}
        t["id"] = t_raw[0]
        t["author_id"] = t_raw[1]
        t["text"] = t_raw[2]
        t["language"] = t_raw[3]
        t["date"] = datetime.strptime(t_raw[4], "%Y-%m-%d %H:%M:%S%z")
        t["tokens"] = json.loads(t_raw[11])
        t["label"] = t_raw[9]
        t["frog"] = json.loads(t_raw[13])
        t["lemmas"] = [term["lemma"] for term in t["frog"]]
        t["annotated"] = t_raw[10] is not None
        
        if t["date"] < dates[0]:
            groups[0].append(t)
        elif t["date"] < dates[1]:
            groups[1].append(t)
        else:
            groups[2].append(t)

    tweets_raw.clear()

    matrix = 0
    for tweets in groups[-2:]:
        # print(tweets)
        # tweets.sort(key = lambda t: t["id"])

        tweets = classifier.feature_extraction(tweets)

        # print("storing tweets")
        # f = open(r"resources\temp.txt", 'w')
        # for tweet in tqdm(tweets):
        #     f.write(json.dumps(tweet) + '\n')
        # f.close()

        print("retrieving tweets")
        # tweets = []
        # for line in tqdm(open(r"resources\temp.txt").read().splitlines()):
        #     tweets.append(json.loads(line))
        tweets_unlabeled = [tweet for tweet in tweets if not tweet["annotated"]]


        # maak een store en retrieve. Maakt het miss ook mogelijk om alpha te testen
        # classified = classifier.classify(tweets, tweets_unlabeled)

        classified = []
        for tweet in tweets_unlabeled:
            if tweet["sim"][0] > tweet["sim"][1]:
                classified.append(0)
            elif tweet["sim"][0] < tweet["sim"][1]:
                classified.append(1)
            else:
                classified.append(2)

        print(classified)

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()


        c.executemany(
            'UPDATE "labels" SET label = ? WHERE id = ?',
            ([(int(classified[i]), tweets_unlabeled[i]["id"]) for i in range(len(classified))]))
        conn.commit()
        conn.close()


def get_results():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute('SELECT * FROM "tweets" INNER JOIN "labels" ON tweets.id = labels.id INNER JOIN "frog_NLTK" ON tweets.id = frog_NLTK.tweet_id')

    tweets_raw = c.fetchall()

    conn.commit()
    conn.close()

    dates = [datetime(2020, 11, 17, tzinfo=timezone.utc), datetime(2020, 12, 8, tzinfo=timezone.utc)]

    tweets = []
    for t_raw in tweets_raw:
        t = {}
        t["id"] = t_raw[0]
        t["text"] = t_raw[2]
        t["date"] = datetime.strptime(t_raw[4], "%Y-%m-%d %H:%M:%S%z")
        t["label"] = t_raw[9]
        t["annotated"] = t_raw[10] is not None
        tweets.append(t)

    # tweets_unannotated = [tweet for tweet in tweets if not tweet["annotated"]]

    sorted_tweets = {}
    for t in tqdm(tweets):
        t_date = t["date"].date()
        if t_date not in sorted_tweets.keys():
            sorted_tweets[t_date] = []
        sorted_tweets[t_date].append(t)

    for k in sorted(sorted_tweets.keys()):
        labels = [tweet["label"] for tweet in sorted_tweets[k]]
        print(f"key: {k} contains: {labels.count(0)} positive, {labels.count(1)} negative, {labels.count(2)} neutral")
        

def test_results():
    #train set
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM "tweets" INNER JOIN "labels" ON tweets.id = labels.id INNER JOIN "frog_NLTK" ON tweets.id = frog_NLTK.tweet_id')
    tweets_raw = c.fetchall()
    conn.commit()
    conn.close()

    tweets = {}
    for t_raw in tweets_raw:
        if t_raw[10] is None:
            tweets[t_raw[0]] = t_raw[9]

    # testset
    conn = sqlite3.connect(r"resources\tweets_test.db")
    c = conn.cursor()
    c.execute('SELECT * FROM "tweets" INNER JOIN "labels" ON tweets.id = labels.id INNER JOIN "frog_NLTK" ON tweets.id = frog_NLTK.tweet_id')
    tweets_raw = c.fetchall()
    conn.commit()
    conn.close()

    test = {}
    for t_raw in tweets_raw:
        if t_raw[10] is not None:
            test[t_raw[0]] = t_raw[9]

    score = np.zeros((3, 3))

    for tweet_id, label in test.items():
        if tweet_id not in tweets.keys():
            print("Too small of a testset")
            continue
        check = tweets[tweet_id]
        score[label][check] += 1        

    print(score)

'''
import sqlite3
DB_FILE = 'resources/tweets_test.db'
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute('SELECT * FROM "tweets" INNER JOIN "labels" ON tweets.id = labels.id INNER JOIN "frog_NLTK" ON tweets.id = frog_NLTK.tweet_id')
tweets = c.fetchall()
tweets_a = [tweet for tweet in tweets if tweet[10] is not None]
labels = [tweet[9] for tweet in tweets_a]
import numpy as np
{i:labels.count(i) for i in np.unique(labels)}
'''

FUNCTION_MAP = {
    's': setup,
    'a': tweet_analysis,
    'l': loop_tweets,
    'p': tweet_filter.process_tweets,
    'c': classifier.annotate,
    'w': classifier.semantic_words,
    'r': get_results,
    't': test_results
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('command', default='s', choices=FUNCTION_MAP.keys())
    args = parser.parse_args()
    func = FUNCTION_MAP[args.command]
    func()
