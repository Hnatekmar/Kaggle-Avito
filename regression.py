import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import KFold
from nltk.stem.snowball import RussianStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import keras
from keras.layers import *
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor

y = pd.read_csv('train.csv')['deal_probability'].fillna(0)

X = pd.read_feather('trainX.pk').fillna(0.0)
X_test = pd.read_feather('testX.pk').fillna(0.0)

kf = KFold(shuffle=True)

clf = keras.Sequential([
    Dense(units=1024, activation='relu', input_dim=X.shape[1]),
    Dropout(0.1),
    Dense(units=512, activation='relu'),
    Dropout(0.1),
    Dense(units=1, activation='sigmoid')
])

clf.compile(loss=keras.losses.mean_squared_error, 
            optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

testData = pd.read_csv('test.csv')

models = [
          ('HuberRegressor', linear_model.HuberRegressor()),
          #('NeuralNet', clf),
          ('Ridge', linear_model.Ridge()),
          ('Lasso', linear_model.Lasso()),
          ('DecisionTreeRegression', DecisionTreeRegressor(max_depth=10))
        ]

frameData = {}
for model_name, clf in models:
    print(model_name)
    scores = []
    for train_index, test_index in kf.split(X):
        clf.fit(X.iloc[train_index], y[train_index]) # Naučení modelu
        scores.append(math.sqrt(mean_squared_error(y[test_index], clf.predict(X.iloc[test_index]))))
    frameData[model_name] = scores

    # Tvorba submission.csv s pomocí modelu
    y_ = clf.predict(X_test)

    testData['deal_probability'] = pd.Series([min(1, max(0, el)) for el in y_]) # Předpověď musí být v rozmezí 0 - 1
    ids = testData[['item_id', 'deal_probability']]
    ids.to_csv(model_name + '_submission.csv', index=False, header=['item_id', 'deal_probability'])
    print(sum(scores) / len(scores))
    print()

frame = pd.DataFrame(frameData)
frame.to_csv('scores2.csv')

