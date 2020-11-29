"""Download and analyse tweets."""
import sqlite3
import argparse
import time
import requests

from datetime import datetime

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
        time.sleep(10)
    
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
    raise NotImplementedError


FUNCTION_MAP = {
    's': setup,
    'a': tweet_analysis,
    'l': loop_tweets
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('command', default='s', choices=FUNCTION_MAP.keys())
    args = parser.parse_args()
    func = FUNCTION_MAP[args.command]
    func()
