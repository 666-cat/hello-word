import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
import numpy as np
import pandas as pd 
from sqlalchemy import false
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
def plot_learning_curve(estimator,title,x,y,ylim,cv=0,n_jobs=1,train_sizes=np.linspace(.1,1.0,5)):
    plt.figure()
    plt.title(title)
    # 修复缺失冒号的问题，同时修正 Not 为 None 比较的正确语法
    if ylim is not None:
        plt.ylim(*ylim)#ylim是一个元组所以加*
    plt.xlabel("training examples")
    plt.ylabel("score")
    train_sizes,train_scores,test_scores=learning_curve(estimator,x,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    print(train_scores)
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="r")
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="g")
    plt.plot(train_sizes,train_scores_mean,'o-',color="r",label="training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color="g",label="cross-validation score")
    plt.legend(loc="best")
    plt.show()
model = DecisionTreeClassifier()
data = pd.read_csv('/Users/weiliangyu/Desktop/diabetes.csv')
X = data.iloc[:, :-1]#提取所有行和除最后一列外的所有列（特征矩阵）。
y = data.iloc[:, -1]
plot_learning_curve(model, 'Learning Curve For DecisionTreeClassifier', X, y, (0.60,1.1), 10)
def plot_confusion_matrix(cm,classes,normalize=False,title='confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    fmt='.2f'if normalize else 'd'
    thresh=cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
        plt.tight_layout()
        plt.ylabel('ture label')
        plt.xlabel('predict label')
    plt.show()
