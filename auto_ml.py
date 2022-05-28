from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pprint
import random
from autogluon.text import TextPredictor

np.random.seed(42)
random.seed(42)

data = pd.read_csv("./autodl-nas/cleaned_reads.csv")
X = data['sequence']
Y = data['copy_number']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

train = pd.DataFrame()
train['seq'] = X_train
train['counts'] = Y_train

test = pd.DataFrame()
test['seq'] = X_test
test['counts'] = Y_test

predictor = TextPredictor(label='counts', path='./autodl-tmp/AutoMl_result',problem_type='regression')
predictor.fit(train)

test_score = predictor.evaluate(test)
print(test_score)