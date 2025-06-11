# 导入NumPy库，用于进行数值计算和数组操作
import numpy as np 
# 导入Matplotlib库的pyplot模块，用于数据可视化
import matplotlib.pyplot as plt 
# 从sklearn库的datasets模块导入make_blobs函数，用于生成聚类数据集
from sklearn.datasets import make_blobs 
# 从sklearn库的model_selection模块导入train_test_split函数，用于划分训练集和测试集
from sklearn.model_selection import train_test_split 

# 定义线性SVM类
class LinearSVM: 
    # 类的构造函数，初始化模型的超参数
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000): 
        """ 
        初始化线性SVM模型 
        
        参数: 
        learning_rate: 学习率，控制梯度下降的步长 
        lambda_param: L2正则化参数，控制模型复杂度 
        n_iters: 最大迭代次数 
        """ 
        # 将学习率赋值给实例属性lr
        self.lr = learning_rate 
        # 将L2正则化参数赋值给实例属性lambda_param
        self.lambda_param = lambda_param 
        # 将最大迭代次数赋值给实例属性n_iters
        self.n_iters = n_iters 
        # 初始化权重向量为None
        self.w = None 
        # 初始化偏置项为None
        self.b = None 
    
    # 训练模型的方法
    def fit(self, X, y): 
        """ 
        使用梯度下降训练SVM模型 
        
        参数: 
        X: 训练数据特征，形状为(n_samples, n_features) 
        y: 训练数据标签，形状为(n_samples,)，标签应为+1或-1 
        """ 
        # 获取训练数据的样本数和特征数
        n_samples, n_features = X.shape 
        
        # 将标签转换为+1和-1，小于等于0的标签转换为-1，大于0的标签转换为+1
        y_ = np.where(y <= 0, -1, 1) 
        
        # 初始化权重向量为全零向量，长度为特征数
        self.w = np.zeros(n_features) 
        print (f"默认权重:{self.w}")
        # 初始化偏置项为0
        self.b = 0 
        
        # 开始梯度下降优化过程，迭代n_iters次
        for _ in range(self.n_iters): 
            # 遍历每个训练样本
            for idx, x_i in enumerate(X): #enumerate 函数返回一个枚举对象，这是一个迭代器，每次迭代会返回一个包含索引和对应元素的元组。
                # 检查是否满足约束条件 y_i(w·x_i + b) >= 1
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1 
                
                if condition: 
                    # 如果满足条件，更新权重（偏置不变），使用L2正则化
                    #print(self.lr,self.lambda_param,self.w)
                    self.w -= self.lr * (2 * self.lambda_param * self.w) 
                    #print (f"权重:{self.w}")
                else: 
                    # 如果不满足条件，同时更新权重和偏置
                    print(self.lr,self.lambda_param,self.w)
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])) 
                    print(x_i,y_[idx])
                    print (f"权重:{self.w}")
                    self.b -= self.lr * y_[idx] 
                    print (f"偏置:{self.b}")
    
    # 预测样本类别的方法
    def predict(self, X): 
        """ 
        预测样本类别 
        
        参数: 
        X: 待预测数据特征，形状为(n_samples, n_features) 
        
        返回: 
        预测标签，形状为(n_samples,)，标签为+1或-1 
        """ 
        # 计算线性输出，即w·X - b
        linear_output = np.dot(X, self.w) - self.b 
        # 使用np.sign函数返回线性输出的符号，作为预测标签
        return np.sign(linear_output) 
    
    # 计算样本到决策边界距离的方法
    def decision_function(self, X): 
        """计算样本到决策边界的距离""" 
        # 计算线性输出，即w·X - b
        return np.dot(X, self.w) - self.b 

# 生成示例数据的函数
def generate_data(): 
    """生成可线性分隔的二维数据集""" 
    # 使用make_blobs函数生成100个样本，分为2类，随机种子为42，聚类标准差为0.6
    X, y = make_blobs(n_samples=10, centers=2, random_state=42, cluster_std=0.6) 
    # 将标签转换为+1和-1，标签为0的转换为-1，标签为1的转换为+1
    y = np.where(y == 0, -1, 1) 
    return X, y 

# 可视化结果的函数
def visualize_results(X, y, model): 
    """可视化SVM分类结果和决策边界""" 
    # 创建一个大小为10x6的图形窗口
    plt.figure(figsize=(10, 6)) 
    
    # 绘制散点图，根据标签y进行颜色编码，使用coolwarm颜色映射，点大小为50，带黑色边框
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k') 
    
    # 获取当前坐标轴对象
    ax = plt.gca() 
    # 获取当前坐标轴的x轴范围
    xlim = ax.get_xlim() 
    # 获取当前坐标轴的y轴范围
    ylim = ax.get_ylim() 
    
    # 在x轴范围上生成30个等间距的点
    xx = np.linspace(xlim[0], xlim[1], 30) 
    # 在y轴范围上生成30个等间距的点
    yy = np.linspace(ylim[0], ylim[1], 30) 
    # 使用meshgrid函数生成二维网格点
    YY, XX = np.meshgrid(yy, xx) 
    # 将二维网格点转换为二维数组
    xy = np.vstack([XX.ravel(), YY.ravel()]).T 
    # 计算每个网格点到决策边界的距离
    Z = model.decision_function(xy).reshape(XX.shape) 
    
    # 绘制决策边界和间隔边界，颜色为黑色，绘制-1、0、1三个水平的等高线，透明度为0.5，线型分别为虚线、实线、虚线
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--']) 
    
    # 找出距离决策边界最近的点，即支持向量
    support_vectors = np.abs(model.decision_function(X)) <= 1.1 
    # 绘制支持向量，点大小为100，无填充颜色，带黑色边框，并添加图例标签
    ax.scatter(X[support_vectors, 0], X[support_vectors, 1], s=100, facecolors='none', edgecolors='k', label='支持向量') 
    
    # 设置图形标题
    plt.title('线性SVM分类结果') 
    # 设置x轴标签
    plt.xlabel('特征1') 
    # 设置y轴标签
    plt.ylabel('特征2') 
    # 显示图例
    plt.legend() 
    # 显示图形
    plt.show() 

# 主函数入口，当脚本作为主程序运行时执行
if __name__ == "__main__": 
    # 调用generate_data函数生成数据集
    X, y = generate_data() 
    print(f"数据集形状: {X.shape}") 
    print(f"标签形状: {y.shape}")
    print(f"数据集内容: {X}")
    print(f"标签内容: {y}")
    # 使用train_test_split函数将数据集划分为训练集和测试集，测试集占比20%，随机种子为42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    # 初始化线性SVM模型，设置学习率为0.01，L2正则化参数为0.01，最大迭代次数为1000
    print("开始lineaesvm")
    model = LinearSVM(learning_rate=0.01, lambda_param=0.01, n_iters=10) 
    # 调用fit方法训练模型
    print("开始fit")
    model.fit(X_train, y_train) 
    
    # 调用predict方法对测试集进行预测
    print("开始predict")
    predictions = model.predict(X_test) 
    # 计算模型的准确率
    accuracy = np.mean(predictions == y_test) 
    # 打印模型准确率，保留两位小数
    print(f"模型准确率: {accuracy * 100:.2f}%") 
    
    # 调用visualize_results函数可视化分类结果
    visualize_results(X, y, model)    