import numpy as np
import pandas as pd
from sklearn import datasets

iris_data = datasets.load_iris() #dictionary -> data and target and features name

#understanding data
# print(iris_data.data)
# print(iris_data.target)
# print(iris_data.feature_names) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


# make dataframe
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
X = df
Y = iris_data.target

# model

from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()
DTC.fit(X, Y)

print(DTC.predict([[2.5, 6.1, 4.5, 8]]))

# show steps of learning on jupyter notebook

from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(DTC, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())