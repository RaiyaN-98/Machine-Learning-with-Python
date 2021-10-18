import numpy as np
import pandas as pd
from pandas.core.indexes.base import InvalidIndexError
from sklearn.tree import DecisionTreeClassifier

## reading data
my_data = pd.read_csv("drug200.csv")
# print(my_data[0:5])

## pre-processing
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
# print(X[0:5])

## As the array contains strings to classify sex, bp, cholesterol we need to conert it to numerical

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_chol = preprocessing.LabelEncoder()
le_chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_chol.transform(X[:,3])

y = my_data["Drug"]

## Setting up the Decision Tree
# spliting the data into test and train sets
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size = .3, random_state=3)

# modeling decision tree
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=10)
drugTree.fit(X_trainset, y_trainset)

# prediction
predTree = drugTree.predict(X_testset)

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

n = len(y_testset)
cn = 0
itt = 0
for i in y_testset:
    if i == predTree[itt]:
        cn += 1
    itt += 1
print("Accuracy :", cn/n)

## Visualization
import pydotplus
import graphviz
import matplotlib.image as mpimg
from sklearn import tree
from six import StringIO
#%matplotlib inline

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out = tree.export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data, class_names=np.unique(y_trainset), filled=True, special_characters=True, rotate=False)
gh = pydotplus.graph_from_dot_data(dot_data.getvalue())   
print(type(gh), type(dot_data))
gh.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
