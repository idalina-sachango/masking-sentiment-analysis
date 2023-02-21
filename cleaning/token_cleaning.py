import nltk
from nltk.tokenize import word_tokenize
import re
import spacy
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")


def clean_tweets(text):
    '''
    Clean the text of tweets
    '''
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text) # websites
    text = re.sub(r"www.\S+", "", text)
    text = re.sub("@[a-z0-9_]+", " ", text) # mentions
    text = re.sub("[^a-z0-9 ]", "", text) # special characters
    
    # remove words with numbers, but with exceptions
    do_not_discard = ['covid19', 'covid-19', 'n95', 'n-95', 'kn95', 'kn-95', 
    'sarscov2', 'sars-cov-2','c19','c-19' ,'cov19','cov-19']
    clean_w = [] 
    text = word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words("english") + ["amp"]
    for w in text:
        if w in stopwords:
            continue
        if w not in do_not_discard:
            w = re.sub("\w*\d\w*", "", w) # token with number in it
            w = re.sub("[^a-z0-9]", " ", w) # special characters
        else:
            w = re.sub("[^a-z0-9]", "", w) # special characters
        if w:
            w = nlp(w)[0].lemma_
            clean_w.append(w)
    return clean_w
