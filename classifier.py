# SVM classfier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC
from nltk import ngrams
from tqdm import tqdm
import numpy as np

import re
import json
import string
from math import log10

NEG_WORDS = [line for line in open(r"resources\negation_words.txt").read().splitlines()]
SENT_TAGS = [line for line in open(r"resources\sentiment_hashtags.txt").read().splitlines()]
PUNCTUATION = string.punctuation


def feature_extraction(tweets):
    # Ngrams
    print("extracting ngrams...")
    ngrams = [set()] * 3
    for tweet in tweets:
        tweet["ngrams"] =  []
        for g in range(3):
            grams = list(ngrams(tweet["tokens"], g + 1))
            gram_count = {}
            for g in grams
                gram_count[g] = gram_count.get(g, 0) + 1

            tweet["ngrams"].append(gram_count)
            ngrams[g].update(tweet["ngrams"][g].keys())

    for tweet in tweets:
        for g in range(3):
            tweet["ngrams"][g] = [tweet["ngrams"][g].get(gram, 0) for gram in ngrams[g]]

    # Negation
    print("extracting negation...")
    for tweet in tweets:
        neg, n_count = False, 0
        for token in tweet["tokens"]:
            if len(token) == '1' and token in PUNCTUATION:
                neg = False
                continue
            if neg:
                token += "_NEG"
                n_count += 1
            if token in NEG_WORDS:
                neg = True
        tweet["negation"] = n_count

    # TODO POS tagger

    # Writing style
    print("extracting writing style...")
    for tweet in tweets:
        # check for repeated alpha characters
        tweet["rep_alpha"] = len([1 for token in tweet["tokens"] if re.search(r'([A-z])\1\1', token)])

        # check for repeated punctuation
        rep_punct = 0
        prev, prev_count = '', 0
        for token in tweet["tokens"]:
            if len(token) == '1' and token in PUNCTUATION:
                if token == prev:
                    if prev_count == 2:
                        prev = ''
                        prev_count = 0
                        rep_punct += 1
                    else:
                        prev_count += 1
                else:
                    prev = token
                    prev_count = 1

        tweet["rep_punct"] = rep_punct

        #check for words all uppercase
        tweet["uppercase"] = len([1 for token in tweet["tokens"] if re.search(r'^[^a-z]*$', token)])

    # TODO Lexicon sentiment

    # Microblogging features
    print("extracting microblogging features...")
    for tweet in tweets:
        # TODO emoticons

        # Sentiment hashtag count
        tweet["sent_tags"] = len([1 for token in tweet["tokens"] if token in SENT_TAGS])

    return tweets


def tf_idf(tweets):
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


def tweet_similarity(tweets):
    print("constructing similarity matrix...")
    s = list()
    tf_idf(tweets)
    for tweet1 in tweets:
        s_ = list()
        for tweet2 in tweets:
            s_.append(cosine_similarity(tweet1["tf_idf"], tweet2["tf_idf"]))
        s.append(s_)

    return s


def classify(tweets_labeled, tweets_unlabeled):
    f_labeled = get_features(tweets_labeled)
    f_unlabeled = get_features(tweets_unlabeled)
    labels = [tweet["label"] for tweet in tweets_labeled]

    # train the model
    print(f"training the model for {len(tweets_labeled[0])} features...")
    lin_clf = LinearSVC()
    lin_clf.fit(f_labeled, labels)

    print("constructing class probability")
    predictions = []
    for i in tqdm(f_unlabeled):
        p = lin_clf.decision_function([i])
        predictions.append(p)
    return predictions


def get_features(tweets):
    features = []
    for tweet in tweets:
        feature = []

        # Ngrams
        for g in range(3):
            feature.extend(tweet["ngrams"][g])

        # Negation
        feature.append(tweet["negation"])

        # Writing style
        feature.append(tweet["rep_alpha"])
        feature.append(tweet["rep_punct"])
        feature.append(tweet["uppercase"])

        # Microblogging features
        feature.append(tweet["sent_tags"])

        features.append(f)

    return features
