import sqlite3
import frog
from tqdm import tqdm

def setup():
    """Create the table."""
    print("Creating table...")
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE "frog_NLTK" (
        "tweet_id"  text,
        "output"    text
        PRIMARY KEY("tweet_id")
            )''')
    conn.commit()
    conn.close()

def store_nltk():
    print("Retreiving tweets...")
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM tweets')
    tweets = c.fetchall()

    print(f"Parsing {len(tweets)} tweets")
    for tweet in tqdm(tweets):
        frog = frog.Frog(
            frog.FrogOptions(tok=False, morph=False, chunking=False, ner=False)
            )
        output = frog.process(tweet[2])

        c.execute(
            'INSERT INTO frog_NLTK (tweet_id, output) VALUES (?,?)',
            (tweet[0], output))
    
    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup()
    store_nltk()
