"""Download and analyse tweets."""
import sqlite3
import argparse
import time
import requests

from datetime import datetime

from resources import config


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
        "language"    text,
        "created_at" DATETIME,
        "tag"   text,
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
        log(str(e))

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
    params = [f'?q=%23{q}&count=1&result_type=recent' for q in TERMS]
    while True:
        for i in range(len(TERMS)):
            params[i] = get_tweets(params[i])
        time.sleep(10)


def tweet_analysis():
    """
    TODO.

    :raises NotImplementedError: [description]
    """
    raise NotImplementedError


FUNCTION_MAP = {
    's': setup,
    'g': get_tweets,
    'a': tweet_analysis,
    'l': loop_tweets
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('command', default='s', choices=FUNCTION_MAP.keys())
    args = parser.parse_args()
    func = FUNCTION_MAP[args.command]
    func()
