# -*- coding: utf-8 -*-

!pip install tabml

!pip install fastFM scikit-learn

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from fastFM import als
import matplotlib.pyplot as plt

import tabml.datasets
df_dict = tabml.datasets.download_movielen_1m()
ratings = df_dict['ratings']

user_id_map = {user_id: idx for idx, user_id in enumerate(ratings['UserID'].unique())}
movie_id_map = {movie_id: idx for idx, movie_id in enumerate(ratings['MovieID'].unique())}

ratings['UserIdx'] = ratings['UserID'].map(user_id_map)
ratings['MovieIdx'] = ratings['MovieID'].map(movie_id_map)

user_features = pd.get_dummies(ratings['UserIdx'], prefix='User')
movie_features = pd.get_dummies(ratings['MovieIdx'], prefix='Movie')

X = pd.concat([user_features, movie_features], axis=1)
y = ratings['Rating'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_sparse = csr_matrix(X_train.values)
X_test_sparse = csr_matrix(X_test.values)

fm = als.FMRegression(n_iter=100, rank=10, random_state=42)

train_losses = []
val_losses = []

for epoch in range(10):
    fm.fit(X_train_sparse, y_train)
    y_pred_train = fm.predict(X_train_sparse)
    y_pred_test = fm.predict(X_test_sparse)

    train_loss = mean_absolute_error(y_train, y_pred_train)
    val_loss = mean_absolute_error(y_test, y_pred_test)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss}, Val Loss = {val_loss}")

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"Train MAE: {mae_train}")
print(f"Test MAE: {mae_test}")

y_true = np.where(y_test >= 3.5, 1, 0)
y_pred = fm.predict(X_test_sparse)

auc = roc_auc_score(y_true, y_pred)
print(f"AUC: {auc}")

y_pred_binary = np.where(y_pred >= 3.5, 1, 0)
f1 = f1_score(y_true, y_pred_binary)
print(f"F1 Score: {f1}")