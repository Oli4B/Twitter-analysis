import sqlite3
import json
import frog
from tqdm import tqdm

DB_FILE = r"resources/tweets.db"

def setup():
    """Create the table."""
    print("Creating table...")
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE "frog_NLTK" (
        "tweet_id"  text,
        "output"    text,
        PRIMARY KEY("tweet_id")
            )''')
    conn.commit()
    conn.close()

def store_nltk():
    print("Retreiving tweets...")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM "tweets"')
    tweets = c.fetchall()

    print(f"Parsing {len(tweets)} tweets")
    for tweet in tqdm(tweets):
        if tweet[2].startswith("RT"):
            continue
        f = frog.Frog(
            frog.FrogOptions(tok=False, morph=False, chunking=False, ner=False)
            )
        output = f.process(tweet[2])

        c.execute(
            'INSERT INTO frog_NLTK (tweet_id, output) VALUES (?,?)',
            (tweet[0], json.dumps(output)))

    print("Done")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup()
    store_nltk()
