import pandas as pd
from pathlib import Path
import re
import nltk
from nltk.stem import WordNetLemmatizer
import gensim 
from gensim import corpora, models 
nltk.download("stopwords")

stopwords = set(nltk.corpus.stopwords.words("english"))

home_path = Path(__file__).parent.parent
data_path = home_path.joinpath("data/")
df = pd.read_csv(data_path.joinpath("covid19_tweets.csv"))

#Filter tweets by "mask"
df = df.loc[df.text.str.contains('mask',case=False)]

stemmer = WordNetLemmatizer()

def pre_process(text, stopwords):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]",' ',text)
    text = text.lower()
    text = text.split()
    text = [stemmer.lemmatize(word) for word in text if len(word) > 3]
    print("text: ", text)
    text_without_sw = [word for word in text if word not in stopwords]
    return text_without_sw

document = df[["text"]]

processed_data = []
for row in document.values:
    text = row[0]
    tokens = pre_process(text, stopwords)
    processed_data.append(tokens)

print(processed_data)

input_dict = corpora.Dictionary(processed_data)
input_corpus = [input_dict.doc2bow(token, allow_update=True) for token in processed_data]
lda_model = gensim.models.ldamodel.LdaModel(input_corpus, num_topics=4, id2word=input_dict, passes=20)
topics = lda_model.print_topics(num_words=10)

for t in topics:
    print(t)

tfidf = models.TfidfModel(input_corpus)
corpus_tfidf = tfidf[input_corpus]
for doc in corpus_tfidf:
    pprint(doc)
    break



