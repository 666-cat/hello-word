# Training Model
## K最近邻算法（K-Nearest Neighbors, K-NN）
   **简介**
   > K-NN 是一种简单直观的机器学习分类与回归算法，属于 ** 惰性学习（Lazy Learning）** 方法，即它不需要在训练阶段构建显式的模型,而是直接在预测时通过对比数据点之间的相似度（距离）来做出决策。

   **核心思想**
   >对于一个待预测的数据点，找到训练数据中与它距离最近的K个邻居，根据这K个邻居的多数类别或平均值(回归问题)来决定该数据点的标签。
   
   **关键要素**
   >1. **距离度量**：通常使用欧氏距离、曼哈顿距离或余弦相似度等。
   >2. **K值选择**：K值的选择对算法的性能有很大影响。通常采用交叉验证或网格搜索等方法来选择最佳的K值。
   >3. **加权**：可以对K个邻居进行加权，距离越近的邻居权重越大。

   **算法流程（以分类问题为例）**
   >1. 计算距离：计算待预测点与训练集中所有样本的距离（如欧氏距离）。
   >2. 选择邻居：按距离从小到大排序，选取前 K 个最邻近的样本。
   >3. 投票分类：统计这 K 个邻居中出现次数最多的类别，作为待预测点的类别。

   **度量距离的方法**
   >1. **欧氏距离**：$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$
   >2. **曼哈顿距离**：$d(x,y) = \sum_{i=1}^{n}|x_i-y_i|$
   >3. **余弦相似度**：$cos(\theta) = \frac{x \cdot y}{||x|| \cdot ||y||}$
   >4. **闵可夫斯基距离**：$d(x,y) = (\sum_{i=1}^{n}|x_i-y_i|^p)^{\frac{1}{p}}$

   **K值选择**
   >1. **交叉验证**：将数据集划分为训练集和验证集，通过在验证集上的性能来选择最佳的K值。
   >2. **网格搜索**：尝试不同的K值，计算每个K值下的性能指标，选择性能最好的K值。

   **优缺点**
   | 优点| 缺点|
   |:---:|:---:|
   |简单直观，无需训练。对非线性数据适应性强|1. 预测时计算量大(需要储存全部数据)。2. 对高维数据效果较差（距离度量失效）。3. 需预处理（如归一化）。|

   **以下是使用 Python 实现 K 最近邻（K-NN）算法的完整代码**

```import numpy as np
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
    print(f"预测结果:{prediction}")```