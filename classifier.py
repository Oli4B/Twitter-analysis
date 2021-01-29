# SVM classfier
from sklearn import svm
from nltk import ngrams
from tqdm import tqdm
import numpy as np
import sqlite3
from multiprocessing import Pool

import os
import re
import time
import json
import string
from math import log10
import random

DB_FILE = r"resources\tweets.db"


def feature_extraction(tweets):
    punctuation = set(string.punctuation)

    # Ngrams
    print("extracting ngrams...")
    ngram_set = [set(), set(), set()]
    n = 2 # set to 2 from 3
    for tweet in tqdm(tweets):
        tweet["ngrams"] =  []
        for i in range(n):
            grams = list(ngrams(tweet["lemmas"], i + 1))
            gram_count = {}
            for g in grams:
                gram_count[g] = gram_count.get(g, 0) + 1

            tweet["ngrams"].append(gram_count)
            ngram_set[i].update(tweet["ngrams"][i].keys())

    print(f"calculating {len(ngram_set[n-1])} ngrams...")
    # for tweet in tqdm(tweets):
        # for g in range(n):
        # tweet["ngrams"][g] = [tweet["ngrams"][g].get(gram, 0) for gram in ngram_set[g]]
    with Pool(os.cpu_count()) as pool:
        tweets = pool.starmap(ngram_helper, [(tweet, n, ngram_set) for tweet in tweets])

    # test = tweets[0]["ngrams"][0]
    # print(f"TEST ngrams: {len(test)}")

    # Negation
    neg_words = [line for line in open(r"resources\negation_words.txt").read().splitlines()]

    print("extracting negation...")
    for tweet in tqdm(tweets):
        neg, n_count = False, 0
        for term in tweet["frog"]:
            if "eos" in term:
                neg = False
                continue
            elif neg:
                term["lemma"] += "_NEG"
                n_count += 1
            elif term["lemma"] in neg_words:
                neg = True
        tweet["negation"] = n_count

    # POS tagger
    print("analyzing POS tags...")
    tagset = set()
    for tweet in tqdm(tweets):
        pos_tags = tweet["frog"]
        tweet["pos"] = {}
        for tag in pos_tags:
            t = tag["pos"].split('(')[0]
            tweet["pos"][t] = tweet["pos"].get(t, 0) + 1
        
        tagset.update(tweet["pos"].keys())

    for tweet in tweets:
        tweet["pos"] = [tweet["pos"].get(p, 0) for p in tagset]

    # Writing style
    triple_punct = [i + i + i for i in punctuation]
    print("extracting writing style...")
    for tweet in tqdm(tweets):
        # check for repeated alpha characters
        tweet["rep_alpha"] = len([1 for token in tweet["tokens"] if re.search(r'([A-z])\1\1', token)])

        # check for repeated punctuation
        rep_punct = 0
        for token in tweet["tokens"]:
            if any([1 for punct in punctuation if punct in token]):
                rep_punct += 1

        tweet["rep_punct"] = rep_punct

        #check for words all uppercase
        tweet["uppercase"] = len([1 for token in tweet["tokens"] if re.search(r'^[^a-z]*$', token)])

    # Lexicon sentiment
    print("extracting lexicon sentiment...")
    sent_words = [line for line in open(r"resources\sentiment_words.txt", encoding="utf-8").read().splitlines()]

    split_index = sent_words.index('')
    pos_words = set(sent_words[:split_index])
    neg_words = set(sent_words[split_index+1:])
    for tweet in tqdm(tweets):
        lex_pos, lex_neg = 0, 0
        for token in tweet["lemmas"]:
            if token in pos_words:
                lex_pos += 1
            if token in neg_words:
                lex_neg += 1
        tweet["sim"] = [lex_pos, lex_neg]

    return tweets

def ngram_helper(tweet, n, ngram_set):
    for g in range(n):
        tweet["ngrams"][g] = [tweet["ngrams"][g].get(gram, 0) for gram in ngram_set[g]]
    return tweet
    

def tf_idf(tweets):
    # not used
    idf = dict()
    for tweet in tweets:
        for token in list(set(tweet["tokens"])):
            idf[token] = idf.get(token, 0) + 1

    idf = {term: log10((len(tweets)+1)/df) for term, df in tweets.items()}

    tf_idf = dict()
    for tweet in tweets:
        tf = dict()
        for token in tweet["tokens"]:
            tf[token] = idf.get(token, 0) + 1

        tweet["tf_idf"] = np.array([[tf.get(t,0)*f for t, f in idf.items()]])


def semantic_words():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM "tweets" INNER JOIN "labels" ON tweets.id = labels.id INNER JOIN "frog_NLTK" ON tweets.id = frog_NLTK.tweet_id')

    tweets_raw = c.fetchall()

    conn.commit()
    conn.close()

    tweets = []
    for t_raw in tweets_raw:
        if t_raw[10] is None:
            continue

        t = {}
        t["id"] = t_raw[0]
        t["lemmas"] = [nltk["lemma"] for nltk in json.loads(t_raw[13])]
        t["label"] = t_raw[9]
        tweets.append(t)

    threshold = 0.6
    min_occurance = 2

    df = [{}, {}, {}]
    terms = set()
    prob_l = np.array([0] * 3)

    # count all document frequencies for each term for each label (0,1,2)
    for tweet in tweets:
        prob_l[tweet["label"]] += 1

        tweet["df"] = set(tweet["lemmas"])
        
        terms.update(tweet["lemmas"])

        for term in tweet["df"]:
            df[tweet["label"]][term] = df[tweet["label"]].get(term, 0) + 1

    # prob_l = np.array([i / sum(prob_l) for i in prob_l])
    words = [[], []]
    
    for term in terms:
        # total number of documents for the term
        total = sum(df[i].get(term, 0) for i in range(3))
        # print(term)
        # print([df[i].get(term, 0) for i in range(3)])
        # print(total)

        if total < min_occurance:
            continue

        # term belongs to the highest prob label
        prob = np.array([label.get(term, 0) for label in df]) / prob_l
        prob = prob / sum(prob)
        max_prob = np.argmax(prob)
        # print(prob)

        # print(f'{term} {[label.get(term, 1) / total for label in df]} {prob}')
        if max_prob != 2 and prob[max_prob] > threshold:
            words[max_prob].append(term)
            # print(f'{term} {prob[max_prob]}')
        # time.sleep(1)

    # print(df)
    # print(words)
    f = open(r"resources\sentiment_words.txt", 'w', encoding="utf-8")
    f.writelines(word + '\n' for word in words[0])
    f.write('\n')
    f.writelines(word + '\n' for word in words[1])
    f.close()


def tweet_similarity(tweets):
    print("constructing similarity matrix...")
    s = list()
    with Pool(os.cpu_count()) as pool:
        s = pool.starmap(sim_helper, [(tweet, tweets) for tweet in tweets])
        # for tweet1 in tqdm(tweets):
        # results = pool.starmap(merge_names, product(names, repeat=2))
        # s_ = sim_helper(tweet1, tweets)
        # s.append(s_)

    return np.array(s)

def sim_helper(tweet1, tweets):
    s_ = list()
    for tweet2 in tweets:
        # sim with itself is 0
        if tweet1 is tweet2:
            s_.append(0)
        else:
            cos_sim = np.dot(tweet1, tweet2)/(np.linalg.norm(tweet1)*np.linalg.norm(tweet2))
            # print(f" {cos_sim} {np.dot(tweet1, tweet2)} {np.linalg.norm(tweet1)} {np.linalg.norm(tweet2)}")
            s_.append(cos_sim if not np.isnan(cos_sim) else 0.)

    return s_


def classify(tweets, tweets_unlabeled):
    tweets_labeled = [tweet for tweet in tweets if tweet["annotated"]]

    print("retrieving features")
    f_labeled = get_features(tweets_labeled)
    f_unlabeled = get_features(tweets_unlabeled)
    labels = [tweet["label"] for tweet in tweets_labeled]

    # train the model
    print(f"training the model for {len(tweets_labeled[0])} features...")
    model = svm.SVC(probability=True)
    model.fit(f_labeled, labels)

    print("constructing initial class probability")
    pi = [[]] * len(f_unlabeled)
    
    # for tweet in tqdm(f_unlabeled):
    #     pi.append(model.predict_proba([tweet]))

    with Pool(os.cpu_count()) as pool:
        pi = pool.starmap(model.predict_proba, [[[f]] for f in f_unlabeled])
    
    tweets_labeled = [tweet for tweet in tweets if tweet["annotated"]]
    tweets_unlabeled = [tweet for tweet in tweets if not tweet["annotated"]]

    # cleanup
    f_labeled.clear()
    f_unlabeled.clear()
    tweets_labeled.clear()
    
    # similarity
    # print([t["sim"] for t in tweets_unlabeled])
    similarity = tweet_similarity([t["sim"] for t in tweets_unlabeled])
    
    # print(similarity)

    # detemine a and I
    a = 0.01
    I = 9

    # C3E algorithm
    print("running C3E")
    N = len(tweets_unlabeled)
    y = np.array([[1/3] * 3] * N)

    for iteration in tqdm(range(I)):
        y_new = np.array([[0] * 3] * N)
        with Pool(os.cpu_count()) as pool:
            y_new = pool.starmap(c3e_helper, [(i, y, similarity, N, a, pi[i]) for i in range(N)])
        
        # for i in range(N):
        #     # print(similarity[i].shape)
        #     # print(y.shape)
        #     # print(pi[i].shape)
        #     h1 = pi[i] + a * np.sum([similarity[i,j] * y[j] for j in range(N)], 0)
        #     h2 = 1 + a * np.sum(similarity[i])
        #     y_new[i] = h1 / h2
        
        y = y_new

        # print([(i[0], np.where(i[0] == max(i[0]))[0][0]) for i in y])
        
    

    return [np.where(i[0] == max(i[0]))[0][0] for i in y]

def c3e_helper(i, y, similarity, N, a, pi):
    h1 = pi + a * np.sum([similarity[i,j] * y[j] for j in range(N)], 0)
    h2 = 1 + a * np.sum(similarity[i])
    r = h1 / h2
    return r


def get_features(tweets):
    features = []
    for tweet in tweets:
        tweet_features = []
        tweet_features.extend(tweet["ngrams"][0])
        tweet_features.extend(tweet["ngrams"][1])
        # tweet_features.extend(tweet["ngrams"][2])

        tweet_features.append(tweet["negation"])

        tweet_features.extend(tweet["pos"])

        tweet_features.append(tweet["rep_alpha"])
        tweet_features.append(tweet["rep_punct"])
        tweet_features.append(tweet["uppercase"])

        tweet_features.extend(tweet["sim"])
        features.append(np.array(tweet_features))

    return features


def annotate(N=20):
    print("Retreiving tweets...")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM "tweets" WHERE EXISTS (SELECT 1 FROM "labels" WHERE tweets.id = labels.id AND labels.annotated IS NULL)')
    
    tweets = c.fetchall()

    print(f"Parsing {len(tweets)} tweets")

    results = {}
    
    i = 0
    while i < N:
        tweet = tweets[random.randint(0,len(tweets))]
        if tweet[0] in results:
            continue

        print(f"{tweet[2]}")

        answer = input("[0,1,2,stop]Enter positive(0), negative(1) or neutral(2): ") 
 
        if answer == "0": 
            results[tweet[0]] = 0
        elif answer == "1": 
            results[tweet[0]] = 1
        elif answer == "2":
            results[tweet[0]] = 2
        elif answer == "stop":
            break

        else: 
            print("Please enter 0, 1 or 2")
 
        i += 1

    print("Inserting into the database")

    c.executemany(
        'UPDATE "labels" SET annotated = ?, label = ? WHERE id = ?',
        ([(1, r[1], r[0]) for r in results.items()]))

    conn.commit()
    conn.close()

    print("Done")
