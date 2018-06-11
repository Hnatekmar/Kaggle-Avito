import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from nltk.stem.snowball import RussianStemmer
from sklearn.preprocessing import LabelEncoder
import _pickle as cPickle
import numpy as np

tf = TfidfVectorizer(analyzer='word', max_features = 100, ngram_range=(1, 1)) # TF-IDF vektorizer. Víceméně každé slovo ohodnotí dle frekvence jeho vyskyu

def column_to_vector(column):    
    return tf.fit_transform(column)

# Přečti a transformuj
dataset = pd.read_csv('train.csv')
dataset = dataset.fillna("")

def prepare(dataset, tf, train, output_y = True):
    """
    Preprocessing pro dataset. X jsou featury (sparse matrix, která vznikne použitím vectorizeru) a y je pravděbodnost toho, že dojde k obchodu (to, co se snažíme předpovídat)
    dataset - Panda dataframe
    tf - vectorizer
    train - určuje, zda se jedná o trénovací data
    output_y - určuje, zda dataset obsahuje deal_probability
    """
    dataset['sum'] = [str(a) + " " + str(b) for a,b in zip(dataset['title'].values, dataset['description'].values)] # Spojení title a description do jednoho sloupce
    dataset['price'] = dataset['price'].map(lambda row: 0 if row == "" else int(row))

    dataset['category'] = [str(a) + " " + str(b) for a,b in zip(dataset['parent_category_name'].values, dataset['category_name'].values)] # Spojení title a description do jednoho sloupce
    dataset['location'] = [str(a) + " " + str(b) for a,b in zip(dataset['region'].values, dataset['city'].values)] # Spojení title a description do jednoho sloupce
    if train:
        vector = tf.fit_transform(dataset['sum'])
    else:
        vector = tf.transform(dataset['sum'])

    df1 = pd.DataFrame(vector.toarray(), columns=tf.get_feature_names())

    X = pd.concat([pd.get_dummies(dataset[['category']]),
                   dataset['price'],
                   df1], axis=1)
    if output_y:
        y = dataset['deal_probability']
        return X, y
    return X

# Příprava dat
X, y = prepare(dataset, tf, True)
X.to_feather('trainX.pk')
    
dataset = pd.read_csv('test.csv')
dataset = dataset.fillna("")
X_test = prepare(dataset, tf, False, False)
X_test.to_feather('testX.pk')
