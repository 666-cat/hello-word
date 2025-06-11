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
    print(f"预测结果:{prediction}")
```
## linear svm(线性支持向量机)
   **简介**
   > 线性支持向量机（Linear Support Vector Machine, L-SVM）是一种用于分类和回归分析的机器学习算法，它的目标是在特征空间中找到一个超平面，将不同类别的数据点分开。

   **核心思想**
   > 线性支持向量机（Linear Support Vector Machine, L-SVM）是一种用于分类和回归分析的机器学习算法，其核心目标是在特征空间中找到一个最优的超平面，将不同类别的数据点尽可能清晰地分开。对于二分类问题，假设数据是线性可分的，线性支持向量机试图找到一个超平面 $w^T x + b = 0$，其中 $w$ 是超平面的法向量，$b$ 是偏置项，$x$ 是数据点的特征向量。
   在众多可能的超平面中，最优超平面是使两个类别之间的间隔（margin）最大的那个。间隔是指超平面到最近的数据点的垂直距离，这些最近的数据点被称为支持向量（Support Vectors）。通过最大化间隔，线性支持向量机可以提高模型的泛化能力，即对未见过的数据的分类准确性。

   **关键要素**
   >- 间隔最大化 ：线性支持向量机的优化目标是最大化间隔。间隔的大小可以通过支持向量到超平面的距离来衡量。数学上，间隔可以表示为 $\frac{2}{||w||}$，其中 $||w||$ 是法向量 $w$ 的范数。因此，优化问题可以转化为在满足分类约束条件下最小化 $\frac{1}{2}||w||^2$。
    >- 分类约束条件 ：对于线性可分的数据，分类约束条件要求所有正类样本满足 $w^T x_i + b \geq 1$，所有负类样本满足 $w^T x_i + b \leq -1$，其中 $x_i$ 是第 $i$ 个样本的特征向量。这些约束条件确保了所有样本都被正确分类，并且支持向量到超平面的距离为 1。
   >- 软间隔（Soft Margin） ：在实际应用中，数据往往不是完全线性可分的。为了处理这种情况，引入了软间隔的概念。软间隔允许一些样本违反分类约束条件，通过引入松弛变量 $\xi_i$ 来衡量每个样本的违反程度。优化问题变为在最小化 $\frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$ 的同时满足 $y_i(w^T x_i + b) \geq 1 - \xi_i$ 和 $\xi_i \geq 0$，其中 $C$ 是惩罚参数，用于控制对违反约束条件的样本的惩罚程度。
   >- 核函数（Kernel Function） ：虽然线性支持向量机只能处理线性可分的数据，但通过核函数可以将数据映射到高维空间，使得在高维空间中数据变得线性可分。常用的核函数包括线性核、多项式核、高斯核（RBF）等。不过，在纯粹的线性支持向量机中，通常使用线性核，即不进行非线性映射。

   **算法流程**
   >- 数据准备
   1.收集和整理训练数据
   2.特征选择和特征缩放
   3.划分训练集和测试集
   >- 模型训练
   1.选择核函数（如线性核、多项式核、RBF 核等）
   2.设置惩罚参数 C 和其他模型参数
   3.求解优化问题，得到最优超平面的参数 w 和 b
   >- 模型评估
   1.使用测试集评估模型性能
   2.计算准确率、精确率、召回率等指标
   3.进行交叉验证，确保模型的泛化能力
   >- 模型应用
   1.使用训练好的模型对新数据进行分类或回归预测
   2.根据预测结果进行业务决策或后续处理
  
   **SVM损失函数**
  \[\min_{w, b} \frac{1}{2} \| w \|^2 + C \sum_{i=1}^{m} \max\left( 0, \, 1 - y_i \left( w^T x_i + b \right) \right)\]

   **SVM核心约束条件**
   |SVM类型|核心约束条件|目标函数优化方向|优化目标|
   |:--:|:--:|:--|:--:|
   |线性可分|\[y_i \left( w^T x_i + b \right) \geq 1\]|最大化硬间隔，无分类误差|\[\min_{w, b} \frac{1}{2} \|w\|^2\]|
   |线性不可分|\[y_i \left( w^T x_i + b \right) \geq 1 - \xi_i \quad (\xi_i \geq 0)\]|平衡间隔与分类误差(软间隔)|\[\min_{w, b, \xi_i} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m \xi_i\]|
   |非线性|\[y_i \left( w^T \phi(x_i) + b \right) \geq 1\]|通过核函数处理非线性可分问题|\[\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m \max\left(0, 1 - y_i \left( w^T \phi(x_i) + b \right)\right)\]|
   
   **代码实现**
   ```import numpy as np
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
    svm.visualize_results(x,y)```
