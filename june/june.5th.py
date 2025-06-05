from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载数据
iris = datasets.load_iris()
print('特征数据:', iris.data)  # 特征数据，形状为 (150, 4)
print('标签数据:', iris.target)  # 标签数据，形状为 (150,)
print('特征名称:', iris.feature_names)  # 特征名称
print('类别名称:', iris.target_names)  # 类别名称
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建决策树（CART，基尼指数）
clf = DecisionTreeClassifier(
    criterion='gini',       # 划分标准（'entropy'为信息增益）
    max_depth=3,            # 最大深度
    min_samples_split=2     # 节点最小样本数
)
clf.fit(X_train, y_train)

# 预测与评估
y_pred = clf.predict(X_test)
print(f"测试集准确率: {clf.score(X_test, y_test):.2f}")

# 可视化树结构
plt.figure(figsize=(12, 8))
plot_tree(clf, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True)
plt.show()

"""
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor

# 加载数据
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建回归树（平方误差）
reg = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=5
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"测试集MSE: {np.mean((y_test - y_pred)**2):.2f}")
"""