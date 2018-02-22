import pandas as pd

from svm import SVMClassifier
import utils


MAX_KGRAM = 16
MIN_KGRAM = 5

REG_PARAM = 0.205


train_files = [
    'data/Xtr0.csv',
    'data/Xtr1.csv',
    'data/Xtr2.csv'
]
train_labels_files = [
    'data/Ytr0.csv',
    'data/Ytr1.csv',
    'data/Ytr2.csv'
]

test_files = [
    'data/Xte0.csv',
    'data/Xte1.csv',
    'data/Xte2.csv'
]

print('Starting...')
train_data = pd.DataFrame()
train_labels = pd.DataFrame()
test_data = pd.DataFrame()

print('Loading data...')
for data_file in train_files:
    train_data = train_data.append(pd.read_csv(data_file, header=None, names=['seq']), ignore_index=True)
for data_file in train_labels_files:
    train_labels = train_labels.append(pd.read_csv(data_file), ignore_index=True)
train_labels = train_labels.drop(train_labels.columns[[0]], axis=1)
for data_file in test_files:
    test_data = test_data.append(pd.read_csv(data_file, header=None, names=['seq']), ignore_index=True)

print('Creating feature space...')
X_train, feature_set = utils.kgram_sparse_matrix(train_data, kgram_level=MAX_KGRAM, kgram_min=MIN_KGRAM)
y_train = train_labels.Bound.values
y_train[y_train==0] = -1
X_test, _ = utils.kgram_sparse_matrix(test_data, kgram_level=MAX_KGRAM, kgram_min=MIN_KGRAM, feature_set=feature_set, increasing=False)

print('Fitting model...')
model = SVMClassifier(reg_alpha=REG_PARAM)
model.fit(X_train, y_train)

print('Predicting...')
predictions = model.predict(X_test)

print('Storing results...')
final = pd.DataFrame(columns=['Id', 'Bound'])
final['Id'] = range(len(predictions))
final['Bound'] = predictions
final['Bound'] = final['Bound'].astype(int)
final['Bound'][final['Bound'] == -1] = 0
final.to_csv('final_results.csv', index=False)

print('Done.')
