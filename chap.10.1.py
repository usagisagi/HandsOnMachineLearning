import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

mnist = fetch_mldata('MNIST original')
scaler = StandardScaler()
x, y = scaler.fit_transform(mnist['data']).astype(np.float32), mnist['target'].astype(np.int32)
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
dnn_clf = tf.contrib.learn.DNNClassifier(
    hidden_units=[300, 100],
    n_classes=10,
    feature_columns=feature_cols)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)

print('train start')
dnn_clf.fit(x_train, y_train, batch_size=50, steps=40000)

y_pred = dnn_clf.predict(x_test)
print(accuracy_score(y_test, y_pred['classes']))