import numpy as np#numpy 提供了大量的数学函数，可对整个数组进行快速运算，无需编写循环（向量化操作）。
from collections import Counter#统计可哈希对象（如字符串、列表、元组）的出现次数，返回一个无序的字典

class KNN:
    def __init__(self, k=3):#默认k等于3，如果没有指定k的值，就会使用默认值3。
        self.k = k
        
    def fit(self, X, y):
        """训练模型（K-NN实际上不进行训练，只需存储数据）"""
        self.X_train = X
        print(f"训练集: {self.X_train}")
        self.y_train = y
        
    def predict(self, X):
        """对新数据进行预测"""
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        """预测单个样本"""
        # 计算距离
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        print(f"距离: {distances}")
        # 获取最近的k个邻居的索引
        k_indices = np.argsort(distances)[:self.k]#argsort()：返回排序后的索引，原数组不变
        print(k_indices)
        print(f"最近的 {self.k} 个邻居索引: {k_indices}")
        # 获取这些邻居的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        print(f"最近的 {self.k} 个邻居标签: {k_nearest_labels}")
        # 投票确定类别
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def _euclidean_distance(self, x1, x2):
        """计算欧氏距离"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

# 示例使用
if __name__ == "__main__":
    # 简单数据集
    X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 4]])
    y_train = np.array([0, 0, 0, 1, 1])
    
    # 初始化模型
    knn = KNN(k=4)
    knn.fit(X_train, y_train)
    
    # 预测新数据点
    X_test = np.array([[3, 2]])
    prediction = knn.predict(X_test)
    print(f"预测结果: {prediction}")  # 输出应为 [0]