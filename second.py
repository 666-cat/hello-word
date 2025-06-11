'''
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
if __name__=="__main__":
    iris=datasets.load_iris()
    x=iris.data
    y=iris.target
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    clf=DecisionTreeClassifier(max_depth=3,criterion="entropy",min_samples_split=2)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    accuracy=clf.score(x_test,y_test)
    print(f"Accuracy:{accuracy*100:.2f}%")
    plot.figure(figsize=(12,8))
    plot_tree(clf,filled=True,feature_names=iris.feature_names,class_names=iris.target_names)
    plot.show()
'''
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

housing=fetch_california_housing()
x=housing.data
y=housing.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
regressor=DecisionTreeRegressor(max_depth=5,criterion="squared_error",min_samples_split=2)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
mse=regressor.score(x_test,y_test)
print(f"Mean Squared Error:{mse:.2f}")
