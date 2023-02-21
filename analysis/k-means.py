from cProfile import label
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import nltk
import nltk
from collections import Counter
from yellowbrick.cluster import KElbowVisualizer
nltk.download('words')

english_vocab = set(w.lower() for w in nltk.corpus.words.words())
keywords = ["mask", "wearamask", "masking", 
            "masked", "unmask", "unmasked",
            "unmasking", "anti-mask", "maskon", 
            "maskoff", "N95", "face cover", "face covering", 
            "face covered", "mouth cover", "mouth covering", 
            "mouth covered", "nose cover", "nose covering", "nose covered", 
            "cloth covering", "cover your face", "coveryourface", "facemask", 
            "face diaper", "n95", "n-95", "kn95", "kn-95", "respirator","covid19", "wear", "people", "face", "do",
            "coronavirus", "covid", "sars"]

home_path = Path(__file__).parent.parent
data_path = home_path.joinpath("data/dailies/")
fig_path = home_path.joinpath("plots/")
cleaning = home_path.joinpath("cleaning/")


def elbow_method(Y_sklearn,number_clusters):
    """
    This is the function used to get optimal number of clusters in order to feed to the k-means clustering algorithm.
    """
    number_clusters = range(1,number_clusters+1)
    kmeans = [KMeans(n_clusters=i, max_iter = 100) for i in number_clusters] # Getting no. of clusters 

    score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))] # Getting score corresponding to each cluster.
    score = [i*-1 for i in score] # Getting list of positive scores.
    
    plt.plot(number_clusters, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Method')
    plt.show()

df = pd.read_pickle(data_path.joinpath("full_tweets_clean.pkl"))

df = pd.DataFrame(df)
df['text'] = df['text'].str.join(" ")


df_copy = df.copy()
print("Making tf-idf vector")
tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english')
print("Initializing dimensionality reduction")
truncate = TruncatedSVD()
print("Making features")
features = tf_idf_vectorizor.fit_transform(df_copy['text'])
print(features.toarray().shape)
print("Done making features")
print("Initializing kmeans")
model = KMeans(n_clusters=7, init='k-means++', max_iter=100, n_init=1)

Y_sklearn = truncate.fit_transform(features)
# visualizer = KElbowVisualizer(model, k=(1,12))
# visualizer.fit(features)
# visualizer.show()
# elbow_method(Y_sklearn, 10)

print("Fitting reduced tf-idf")
fit = model.fit(Y_sklearn)
print("Predicting clusters from tf-idf")
predict = model.predict(Y_sklearn)
df_copy['cluster'] = model.labels_
df_copy['text'] = df_copy['text'].str.split()
   

def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        print(sorted_means)
        features = tf_idf_vectorizor.get_feature_names_out()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

     


fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predict,cmap='viridis') # Plotting scatter plot 
ax.legend(*sc.legend_elements(), title='clusters')

centers2 = fit.cluster_centers_ # It will give best possible coordinates of cluster center after fitting k-means

ax.scatter(centers2[:, 0], centers2[:, 1],c='black', s=300, alpha=0.6)
ax.set_xlabel('principal component 1 (tf-idf score)')
ax.set_ylabel('principal component 2 (tf-idf score)')
plt.savefig(fig_path.joinpath("cluster-unscaled.png"))


top_clusters = get_top_features_cluster(features.toarray(), predict, 15)

for idx, df in enumerate(top_clusters):
    df.sort_values('score')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(df.iloc[:,0], df.iloc[:,1], align='center')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('tf-idf score')
    ax.set_title(f'Cluster {idx+1}')
    plt.savefig(fig_path.joinpath(f'cluster-words-{idx+1}.png'))
