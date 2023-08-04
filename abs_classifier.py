"""
Sentence classifier to detect absorption.

Usage (example):
    python abs_classifier.py filename.csv wiki_unigrams.bin
"""

import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer 
import pandas as pd
import sklearn
import numpy as np

import imblearn.pipeline as imb
from imblearn.under_sampling import RandomUnderSampler 

#https://github.com/epfml/sent2vec
import sent2vec 

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



# Input .csv file must have the sentence in the first column and the class ("nonabs" or "abs") in the fourth column
inf = sys.argv[1]

# Pre-trained sentence embeddings
modfile = sys.argv[2]

df = pd.read_csv(inf, sep=',', names=["text", "pol", "subj", "gold"])


''' Preprocessing: lowercasing, punctuation removal, stemming '''
df['text'] = df['text'].apply(lambda x: " ".join(i.lower() for i in x.split()))
df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'] = df['text'].apply(lambda x: " ".join(i for i in x.split() if not i.isdigit()))

df['stems'] = df['text'].apply(lambda x: " ".join( PorterStemmer().stem(i) for i in x.split() ) )
df['word_counts']=df['stems'].apply(lambda x: len(str(x).split(" ")))


''' generate a sentence embedding vector '''
model = sent2vec.Sent2vecModel()
model.load_model(modfile)
df['emb'] = df['text'].apply(lambda x: np.array(model.embed_sentence(x)))


''' define the features for classification '''
X = np.vstack(df['emb'])


''' define the classes '''
y = df.iloc[:,3]
print("\nClasses:", y.unique())
print()

''' train and test in k-fold setup '''
kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
score_array = []
for train_index, test_index in kf.split(X, y):
    # Train-Test-Split with Stratified Cross Validation
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    ''' one of these vectorizers apply when X is textual '''
    #vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = (1,1), min_df=2,  max_features=4000)          
    vectorizer = CountVectorizer()       

    ''' make pipeline using undersampling at training time, either with LR or RF '''
    #clf = imb.make_pipeline(RandomUnderSampler(),LogisticRegression(class_weight='balanced') )
    clf = imb.make_pipeline(RandomUnderSampler(),
                            RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)) 
    
    ''' no undersampling '''
    #clf = imb.make_pipeline(LogisticRegression(class_weight='balanced'))

    '''train '''
    clf.fit(X_train, y_train)     
    '''test '''
    y_pred = clf.predict(X_test)     
    ''' evaluate per fold '''
    print(classification_report(y_test,y_pred))
    score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

''' take the mean of the k-fold eval for each metric '''
np.set_printoptions(precision=3, suppress=True)
avg_score = np.mean(score_array, axis=0)
print('P, R, F, support' )
print(avg_score)


