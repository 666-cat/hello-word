import numpy as np
import matplotlib.pyplot as plot
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
class linearSVM:
    def __init__(self,learing_rate=0.001,lambda_param=0.01,n_iters=1000):
        self.lr=learing_rate
        self.lp=lambda_param
        self.n_iters=n_iters
        self.w=None
        self.b=None
    def fit (self,x,y):
        y_=np.where(y<=0,-1,1)
        n_samples,n_features=x.shape
        self.w=np.zeros(n_features)
        self.b=0
        for _ in range(self.n_iters):
            for idx,x_i in enumerate(x):
                condition=y_[idx]*(np.dot(x_i,self.w)-self.b)>=1
                if condition:
                    self.w-=self.lr*(2*self.lp*self.w)
                else:
                    self.w-=self.lr*(2*self.lp*self.w-np.dot(y_[idx],x_i))
                    self.b-=self.lr*y_[idx]
    def predict(self,x):
        outcom=np.dot(x,self.w)-self.b
        return np.sign(outcom)
    def accuracy(self,y_ture,y_predict):
        accuracy=np.mean(y_ture==y_predict)
        return accuracy
    def generate_data(self,n_samples=100,centers=2,random_state=42):
        x,y=make_blobs(n_samples=n_samples,centers=centers,random_state=random_state)#X 是数据集， y 是对应的标签，表示每个样本所属的簇。
        y=np.where(y==0,-1,1)
        return x,y
    def decision_function(self, x):
        return np.dot(x, self.w) - self.b
    def visualize_results(self,x,y):
        plot.figure(figsize=(10,6))
        plot.scatter(x[:,0],x[:,1],c=y,cmap="coolwarm",s=30,edgecolors="k")
        ax=plot.gca()
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        xx=np.linspace(xlim[0],xlim[1],50)
        yy=np.linspace(ylim[0],ylim[1],50)
        XX,YY=np.meshgrid(xx,yy)#XX 数组的每一行元素都相同，且等于 xx 。YY 数组的每一列元素都相同，且等于 yy 。
        xy=np.vstack([XX.ravel(),YY.ravel()]).T#np.vstack 函数将 XX.ravel() 和 YY.ravel() 沿着垂直方向堆叠在一起，形成一个新的数组。ravel 函数将数组展平成一维数组。
        z=self.decision_function(xy).reshape(XX.shape)#
        ax.contour(XX,YY,z,colors="k",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
        support_vectors=np.abs(self.decision_function(x))<=1
        ax.scatter(x[support_vectors,0],x[support_vectors,1],s=100,facecolors="none",edgecolors="k")
        ax.set_title("Linear SVM Decision Boundary")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        plot.legend()
        plot.show()
if __name__=="__main__":
    svm=linearSVM()
    x,y=svm.generate_data()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    svm.fit(x_train,y_train)
    predict=svm.predict(x_test)
    accuracy=svm.accuracy(y_test,predict)
    print(f"准确率:{accuracy*100:.2f}%")
    '''
    outcom=svm.predict(np.array([[float(input("请输入你要预测的x1")),float(input("请输入你要预测的x2"))]]))
    print(f"预测结果：{outcom}")'''
    svm.visualize_results(x,y)

