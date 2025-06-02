import numpy as np
from collections import Counter
class KNM:
    def __init__(self,k=3):
        self.k=k
    def fit (self,x,y):
        self.x_train=x
        self.y_train=y
    def _distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    def _predict(self,x):
        distances=[self._distance(x,x_train) for x_train in self.x_train]
        #print (f"距离:{distances}")
        indices=np.argsort(distances)[:self.k]
        #print (f"最近的{self.k}个邻居索引:{indices}")
        nearest_labels=[y_train[i] for i in indices]
        lastresult=Counter(nearest_labels).most_common(1)
        return lastresult[0][0]
    def predict(self,t):
        y_pred=[self._predict(x) for x in t]
        return np.array(y_pred)
if __name__=="__main__":
    x_train=np.array([[1,2],[2,3],[3,1],[4,3],[5,4]])
    y_train=np.array([0,0,0,1,1])
    knm=KNM(k=4)
    knm.fit(x_train,y_train)
    x_test=np.array([[float(input("请输入测试数据的第一个指标:")),float(input("请输入测试数据的第二个指标:"))]])
    prediction=knm.predict(x_test)
    print(f"预测结果:{prediction}")