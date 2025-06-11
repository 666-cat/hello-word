import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# =============
# 1. 手动实现RBF SVM（简化版）
# =============
class RBFSVM:
    def __init__(self, C=1.0, gamma=0.1, max_iter=1000):
        self.C = C          # 正则化参数
        self.gamma = gamma  # RBF核参数
        self.max_iter = max_iter  # 最大迭代次数
        self.alphas = None  # 拉格朗日乘子
        self.b = None       # 偏置项
        self.support_vectors = None  # 支持向量
        self.support_vector_labels = None  # 支持向量标签
    
    def rbf_kernel(self, X1, X2):
        """计算RBF核函数"""
        n_samples1 = X1.shape[0]
        n_samples2 = X2.shape[0]
        kernel = np.zeros((n_samples1, n_samples2))
        
        # 计算所有样本对的欧氏距离平方
        for i in range(n_samples1):
            for j in range(n_samples2):
                diff = X1[i] - X2[j]
                kernel[i, j] = np.exp(-self.gamma * np.dot(diff, diff))
        return kernel
    
    def fit(self, X, y):
        """训练RBF SVM模型（简化版SMO算法）"""
        n_samples, n_features = X.shape
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # 标签转换：0和1 -> -1和1
        y = np.where(y == 0, -1, 1)
        
        for _ in range(self.max_iter):
            alphas_changed = 0
            for i in range(n_samples):
                # 计算预测值
                kernel_matrix = self.rbf_kernel(X, X[[i]])
                f_x_i = np.sum(self.alphas * y * kernel_matrix.flatten()) + self.b
                
                # 检查KKT条件
                if (y[i] * f_x_i < 1 and self.alphas[i] < self.C) or \
                   (y[i] * f_x_i > 1 and self.alphas[i] > 0):
                    
                    # 随机选择第二个alpha
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # 计算第二个样本的预测值
                    kernel_matrix_j = self.rbf_kernel(X, X[[j]])
                    f_x_j = np.sum(self.alphas * y * kernel_matrix_j.flatten()) + self.b
                    
                    # 保存旧的alphas
                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]
                    
                    # 计算上下界
                    if y[i] != y[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)
                    
                    if L == H:
                        continue
                    
                    # 计算核函数值
                    K_ii = self.rbf_kernel(X[[i]], X[[i]])[0, 0]
                    K_jj = self.rbf_kernel(X[[j]], X[[j]])[0, 0]
                    K_ij = self.rbf_kernel(X[[i]], X[[j]])[0, 0]
                    
                    # 计算eta
                    eta = 2 * K_ij - K_ii - K_jj
                    if eta >= 0:
                        continue
                    
                    # 更新alpha_j
                    self.alphas[j] -= y[j] * (f_x_i - y[i] - (f_x_j - y[j])) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # 更新alpha_i
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    # 计算偏置b
                    b1 = self.b - (f_x_i - y[i]) - y[i] * K_ii * (self.alphas[i] - alpha_i_old) - \
                         y[j] * K_ij * (self.alphas[j] - alpha_j_old)
                    b2 = self.b - (f_x_j - y[j]) - y[i] * K_ij * (self.alphas[i] - alpha_i_old) - \
                         y[j] * K_jj * (self.alphas[j] - alpha_j_old)
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    alphas_changed += 1
            
            # 如果没有alpha被更新，则收敛
            if alphas_changed == 0:
                break
        
        # 保存支持向量
        sv_mask = self.alphas > 1e-5
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.alphas = self.alphas[sv_mask]
    
    def predict(self, X):
        """预测函数"""
        kernel_matrix = self.rbf_kernel(X, self.support_vectors)
        y_pred = np.sum(self.alphas * self.support_vector_labels * kernel_matrix, axis=1) + self.b
        return np.where(y_pred >= 0, 1, 0)  # 转换回0和1

# =============
# 2. 使用scikit-learn实现RBF SVM
# =============
def sklearn_rbf_svm(X_train, X_test, y_train, y_test):
    """使用sklearn实现RBF SVM并进行参数调优"""
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 参数网格搜索
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1]
    }
    
    # 创建SVM分类器
    svm = SVC(kernel='rbf')
    
    # 网格搜索
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    # 获取最佳模型
    best_svm = grid_search.best_estimator_
    print(f"最佳参数: C={best_svm.C}, gamma={best_svm.gamma}")
    
    # 预测并评估
    y_pred = best_svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    
    return best_svm, X_train_scaled, X_test_scaled

# =============
# 3. 可视化函数
# =============
def plot_decision_boundary(X, y, model, title="决策边界"):
    """绘制决策边界"""
    h = 0.02  # 网格步长
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    
    # 计算网格范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()

# =============
# 4. 主函数：生成数据并测试模型
# =============
def main():
    # 生成非线性可分数据（环形数据）
    X, y = datasets.make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 使用手动实现的RBF SVM
    print("===== 手动实现的RBF SVM =====")
    manual_svm = RBFSVM(C=1.0, gamma=0.1)
    manual_svm.fit(X_train, y_train)
    y_pred_manual = manual_svm.predict(X_test)
    accuracy_manual = accuracy_score(y_test, y_pred_manual)
    print(f"手动实现的测试集准确率: {accuracy_manual:.4f}")
    
    # 可视化手动实现的决策边界
    plot_decision_boundary(X_train, y_train, manual_svm, "手动实现RBF SVM的决策边界")
    
    # 使用sklearn实现的RBF SVM
    print("\n===== sklearn实现的RBF SVM =====")
    best_svm, X_train_scaled, X_test_scaled = sklearn_rbf_svm(X_train, X_test, y_train, y_test)
    
    # 可视化sklearn实现的决策边界
    plot_decision_boundary(X_train_scaled, y_train, best_svm, "sklearn RBF SVM的决策边界")

if __name__ == "__main__":
    main()