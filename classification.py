import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from nltk.stem.snowball import RussianStemmer

tf = TfidfVectorizer(analyzer='word', ngram_range=(5, 5)) # TF-IDF vektorizer. Víceméně každé slovo ohodnotí dle frekvence jeho vyskyu
def column_to_vector(column):    
    return tf.fit_transform(column)

# Přečti a transformuj
dataset = pd.read_csv('train2.csv')
dataset = dataset.fillna("")

def prepare(dataset, tf, train, output_y = True):
    """
    Preprocessing pro dataset. X jsou featury (sparse matrix, která vznikne použitím vectorizeru) a y je pravděbodnost toho, že dojde k obchodu (to, co se snažíme předpovídat)
    dataset - Panda dataframe
    tf - vectorizer
    train - určuje, zda se jedná o trénovací data
    output_y - určuje, zda dataset obsahuje deal_probability
    """
    stemmer = RussianStemmer(ignore_stopwords=True) # Provádí stemming tedy převedení slova na jeho kořen. Redukuje počet feature (slov), které musíme brát v potaz. Také odstraňuje častě vyskytující se slovat (stopwords)
    dataset['sum'] = [stemmer.stem(str(a) + " " + str(b)) for a,b in zip(dataset['title'].values, dataset['description'].values)] # Spojení title a description do jednoho sloupce
    if train:
        vector = tf.fit_transform(dataset['sum'])
    else:
        vector = tf.transform(dataset['sum'])

    X = vector
    if output_y:
        y = pd.cut(dataset['deal_probability'], 2000, labels=False) # BINNING (rozřeže deal_probability na 3 částí)
        return X, y
    return X

# Příprava dat
X, y = prepare(dataset, tf, True)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import sklearn

models = [
#        ("LogisticRegression", LogisticRegression()),
        ("DecisionTree", tree.DecisionTreeClassifier(criterion = 'entropy'))
        ]
for name, clf in models:
    kf = KFold(n_splits = 10)
    for train_index, test_index in kf.split(X):
        clf.fit(X[train_index], y[train_index]) # Naučení modelu
    print('"' + name + '",', accuracy_score(y, clf.predict(X)))

    # Tvorba submission.csv s pomocí modelu
    testData = pd.read_csv('test.csv')
    X = prepare(testData, tf, False, False)
    y = clf.predict(X)
    y = [el / 2000 + 5 / 20000 for el in y] # Převedeme třídu na procenta. Jelikož jsme rozsah (0.0 - 1.0) rozdělili na 100 dílu každá třída odpovídá jednomu procentu. Podělíme jí tedy 100 (například z 1 se stane 0.01) a přidáme půl procenta (0.005)
    testData['deal_probability'] = pd.Series(y) # Předpověď musí být v rozmezí 0 - 1
    ids = testData[['item_id', 'deal_probability']]
    ids.to_csv(name + '_submission.csv', index=False, header=['item_id', 'deal_probability'])
