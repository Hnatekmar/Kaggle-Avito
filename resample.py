#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from random import sample
import numpy as np

dataset = pd.read_csv('train.csv')

mean = 126144
while True:
    zero_index = dataset.index[dataset['deal_probability'].between(-1.0, 0.1, inclusive=True)].tolist()
    i = str(len(zero_index))
    drop_count = len(zero_index) - mean
    if drop_count == 0:
        break
    print(i + ' vs ' + str(mean))
    dataset = dataset.drop(sample(zero_index, drop_count))


zero_index = dataset['deal_probability'].between(-0.01, 0.1, inclusive=True).index
print('Deal length: ', zero_index.shape)

dataset['deal_probability'].hist(bins=10)
plt.show()


print(dataset['deal_probability'].value_counts(bins=10))

dataset.to_csv('downsampled.csv')

