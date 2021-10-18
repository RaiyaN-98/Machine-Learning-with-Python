import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

mydata = pd.read_csv("ChurnData.csv")

mydata = mydata[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
mydata['churn'] = mydata['churn'].astype(int)

X = np.asarray(mydata[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(mydata['churn'])

# Normalization
X = preprocessing.StandardScaler().fit(X).transform(X)

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Modeling
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

# predict
yhat = LR.predict(X_test)

# probability of all the classes
yhat_prob = LR.predict_proba(X_test)

## Evaluation
# jaccard index
from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat))

# f1_score
from sklearn.metrics import f1_score
print("F1 score: ", f1_score(y_test, yhat))

# computing logloss
from sklearn.metrics import log_loss
print("Log loss: ", log_loss(y_test, yhat_prob))

# confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matix, without normalization")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# compute confusion matix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision = 2)

# classification_report
print(classification_report(y_test, yhat))

# plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = ['churn=1', 'churn=0'])
plt.show()

