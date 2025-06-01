import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import graphviz 
from sklearn import model_selection
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifie
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings
warnings.filterwarnings("ignore")
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,#estimator：训练的模型对象，如 LinearRegression()、RandomForestClassifier() 等。title：图表的标题。X：特征数据（输入变量）。y：目标数据（输出变量、标签）ylim：(ymin, ymax)，设置 y 轴的显示范围。如果为 None，则自动设置。cv：交叉验证的拆分策略。例如可以是 5（5折交叉验证），也可以是 KFold() 对象。
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):#n_jobs：并行运行的数量。-1 表示使用所有CPU核。train_sizes：指定训练集大小的比例序列。这里是 np.linspace(0.1, 1.0, 5)，表示从10%到100%的训练数据分成5份。
    """
    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html
    """
    plt.figure()#创建一个新的图像窗口。
    plt.title(title)#设置标题。
    if ylim is not None:#如果传入了 ylim，则设定 y 轴的取值范围。
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")#设置 x 轴和 y 轴的标签。
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)#调用 sklearn.model_selection.learning_curve 函数，返回：train_sizes：实际使用的训练样本数。train_scores：每个训练集大小下的训练得分（交叉验证的多个得分）。test_scores：每个训练集大小下的交叉验证得分。
    train_scores_mean = np.mean(train_scores, axis=1)#mean是平均值，axis=1表示按行求平均值。
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()#添加网格线，帮助阅读
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,#使用 fill_between 画出训练得分和测试得分的标准差范围（“阴影带”）。
                     train_scores_mean + train_scores_std, alpha=0.1,#alpha=0.1 控制透明度。
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",#使用折线图绘制平均训练得分和交叉验证得分的曲线。'o-' 表示带圆点的线。
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")#显示图例，自动选择最合适的位置。
    return plt

def plot_confusion_matrix(cm, classes,#cm：混淆矩阵，是一个二维 numpy 数组（可以用 sklearn.metrics.confusion_matrix() 生成）。classes：类别名称列表，用于标签坐标轴（如 [‘cat’, ‘dog’, ‘rabbit’]）。
                          normalize=False,#normalize：是否对混淆矩阵进行归一化处理（将计数转换为百分比）。默认不归一化。
                          title='Confusion matrix',#title：图表标题
                          cmap=plt.cm.Blues):#cmap：颜色映射方案（colormap），默认使用蓝色系（plt.cm.Blues）。
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)#将混淆矩阵 cm 渲染成图像。interpolation='nearest' 保证图像像素清晰无插值。cmap=cmap 设置颜色样式（默认为蓝色渐变）。
    plt.title(title)#title 设置图像的标题。
    plt.colorbar()#colorbar() 显示图像右侧的颜色标尺，表示不同颜色对应的数值范围。
    tick_marks = np.arange(len(classes))#tick_marks 为类别的坐标索引。
    plt.xticks(tick_marks, classes, rotation=45)#xticks() 和 yticks() 分别设置 x 和 y 轴的标签为类别名。rotation=45 让 x 轴标签倾斜 45°，便于阅读
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'#若启用 normalize=True，则数值格式为小数（保留两位）；否则为整数（如样本个数）。
    thresh = cm.max() / 2.#阈值设为最大值的一半。用于决定单元格中文字显示为白色或黑色（便于阅读）。
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):#itertools.product() 生成所有单元格的 (i, j) 坐标。
        plt.text(j, i, format(cm[i, j], fmt),#在每个单元格内添加文字（即预测数量或比例）。
                 horizontalalignment="center",#horizontalalignment="center" 让文字居中显示。
                 color="white" if cm[i, j] > thresh else "black")#根据阈值改变文字颜色（深色格用白字，浅色格用黑字）。
    plt.tight_layout()#tight_layout() 自动调整图像元素，避免重叠。
    plt.ylabel('True label')
    plt.xlabel('Predicted label')#设置 y 轴为“真实标签”，x 轴为“预测标签”。

def compareABunchOfDifferentModelsAccuracy(a, b):#a: 特征数据集（通常是训练集 X_train）。b: 标签数据集（通常是训练集对应的标签 y_train）。函数目标：评估多种分类模型在数据集 (a, b) 上的准确率表现。
    """
    compare performance of classifiers on X_train, X_test, Y_train, Y_test
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
    """    
    print('\nCompare Multiple Classifiers: \n')
    print('K-Fold Cross-Validation Accuracy: \n')
    names = []#保存模型名称（如 'LR', 'RF' 等）。
    models = []#保存模型元组（名称 + 实例）。
    resultsAccuracy = []#保存每个模型在交叉验证中的准确率结果。
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('LSVM', LinearSVC()))
    models.append(('GNB', GaussianNB()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    for name, model in models:
        model.fit(a, b)# 用训练数据 (a, b) 拟合模型
        kfold = model_selection.KFold(n_splits=10, shuffle=True,random_state=7) # 创建10折交叉验证分割器，shuffle=True表示每次分割前打乱数据
        accuracy_results = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')# 执行交叉验证：在训练数据 (a, b) 上进行10折验证，计算准确率：会输出array([0.78, 0.80, 0.76, 0.82, 0.77, 0.79, 0.81, 0.75, 0.80, 0.78])
        resultsAccuracy.append(accuracy_results)
        names.append(name)
        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(), accuracy_results.std()) # 打印模型名称、平均准确率和标准差
        print(accuracyMessage) 
    # Boxplot
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison: Accuracy') # 总标题
    ax = fig.add_subplot(111) # 1行1列第1个子图
    plt.boxplot(resultsAccuracy) #绘制箱线图：展示各模型准确率的分布
    ax.set_xticklabels(names)
    ax.set_ylabel('Cross-Validation: Accuracy Score') # 设置X轴标签为模型名称，Y轴标签为准确率
    plt.show()    
      
def defineModels():
    print('\nLR = LogisticRegression')
    print('RF = RandomForestClassifier')
    print('KNN = KNeighborsClassifier')
    print('SVM = Support Vector Machine SVC')
    print('LSVM = LinearSVC')
    print('GNB = GaussianNB')
    print('DTC = DecisionTreeClassifier')
    print('GBC = GradientBoostingClassifier \n\n')

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "MLPClassifier", "AdaBoost",
         "Naive Bayes", "QDA"]
"""
1. Nearest Neighbors（K最近邻算法）原理：通过计算未知样本与已知样本的距离，取距离最近的K个样本，根据它们的类别投票决定未知样本的类别。特点：简单直观，无需训练模型，但计算量大，对高维数据效果差。
2. Linear SVM（线性支持向量机）原理：寻找一个线性超平面，最大化不同类别样本之间的间隔，适用于线性可分的数据。特点：抗噪声能力强，适合高维特征，但无法处理非线性关系。
3. RBF SVM（径向基函数SVM）原理：使用径向基函数（如高斯函数）作为核函数，将低维非线性数据映射到高维空间，转化为线性可分问题。特点：可处理非线性数据，灵活性高，但参数调优较复杂。
4. Gaussian Process（高斯过程）原理：基于概率模型，假设数据服从高斯分布，通过先验分布和观测数据推断未知样本的概率分布。特点：可给出预测的不确定性，适合小样本学习，但计算复杂度高。
5. Decision Tree（决策树）原理：通过递归划分特征空间，构建树形结构，每个节点基于特征阈值分裂，叶节点为类别标签。特点：可解释性强，能自动处理特征交互，但易过拟合（尤其深度大时）。
6. Random Forest（随机森林）原理：集成多个决策树，通过随机采样样本和特征构建树，最终通过投票或平均输出结果。特点：抗过拟合能力强，鲁棒性好，是常用的“开箱即用”算法。
7. MLPClassifier（多层感知机，神经网络）原理：由多个神经元层组成的神经网络，通过非线性激活函数处理特征，学习复杂的映射关系。特点：适合处理高维、非线性数据，但需大量数据和计算资源，易过拟合。
8.AdaBoost（自适应增强算法）原理：集成多个弱分类器，通过迭代调整样本权重（错分样本权重更高），最终组合成强分类器。特点：对噪声和异常值较敏感，但精度通常较高。
9. Naive Bayes（朴素贝叶斯）原理：基于贝叶斯定理，假设特征之间相互独立，计算样本属于各分类的后验概率。特点：计算效率高，适合文本分类等场景，但“特征独立”假设在现实中常不成立。
10. QDA（二次判别分析）原理：与线性判别分析（LDA）类似，但假设不同类别数据的协方差矩阵不同，通过二次函数划分决策边界。特点：适用于非线性可分数据，但需更多样本避免过拟合。
"""

classifiers = [
    KNeighborsClassifier(),#K 最近邻分类器，基于距离找最近的 K 个点进行投票。适合小数据集。
    SVC(kernel="linear"),#支持向量机（SVM）线性核，用于线性可分的数据
    SVC(kernel="rbf"),#SVM 的高斯径向基核（RBF），适合非线性问题。
    GaussianProcessClassifier(),#高斯过程分类器，适合复杂非线性问题，但计算量大
    DecisionTreeClassifier(),#决策树分类器，易于解释，但可能过拟合。
    RandomForestClassifier(),#随机森林（多个决策树的集成），稳定性好，抗过拟合能力强。
    MLPClassifier(),#多层感知机（神经网络），支持非线性，适合特征复杂的分类问题。
    AdaBoostClassifier(),#自适应提升（集成学习方法），结合弱分类器形成强分类器，常用于提升简单模型效果。
    GaussianNB(),#朴素贝叶斯分类器，假设特征独立，适合文本分类、初始模型尝试。
    QuadraticDiscriminantAnalysis()#二次判别分析，基于贝叶斯判别规则，假设类别服从不同协方差的高斯分布。
]

dict_characters = {0: 'Healthy', 1: 'Diabetes'}
dataset = read_csv('/Users/weiliangyu/Desktop/diabetes.csv')
dataset.head(10)
print("Dataset type: ", type(dataset))
print(dataset.head(10))
def plotHistogram(values,label,feature,title):#values:数据类型必须是DataFrame；label：数据类型必须是str/Series；并且feature和label必须在values中.
    sns.set_style("whitegrid")# 设置Seaborn图表风格为白底网格
    plotOne = sns.FacetGrid(values, hue=label,aspect=2)#创建分面网格，按label列分组（健康/糖尿病，颜色：由 hue 参数分组，默认使用 deep 调色板的前两种颜色，可通过 palette 参数自定义
    plotOne.map(sns.distplot,feature,kde=False)# 在每个分组内绘制feature的直方图（不显示核密度估计）组距：默认使用 Freedman-Diaconis 规则自动计算，可通过 bins 参数自定义
    plotOne.set(xlim=(0, values[feature].max()))# 设置x轴范围从0到该特征的最大值
    plotOne.add_legend()# 添加图例区分不同分组
    plotOne.set_axis_labels(feature, 'Proportion')# 设置坐标轴标签
    plotOne.fig.suptitle(title)# 设置图表标题
    plt.show() # 显示图表

plotHistogram(dataset,"Outcome",'Insulin','Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(dataset,"Outcome",'SkinThickness','SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
dataset2 = dataset.iloc[:, :-1]#从原始数据集 dataset 中提取所有行（:）和除最后一列外的所有列（:-1）。
print("# of Rows, # of Columns: ",dataset2.shape)#输出 dataset2 的维度信息。
print("\nColumn Name           % of Null Values\n")#输出格式化表头，用于展示列名和零值数量的对应关系。
print(((dataset2[:] == 0).sum())/768*100)#计算 dataset2 中每一列中零值的比例，并乘以 100 得到百分比形式。
g = sns.heatmap(dataset.corr(),cmap="BrBG",annot=False)#dataset.corr()：计算相关系数矩阵，sns.heatmap()：绘制热力图，cmap="BrBG"：使用蓝色-白色-棕色的颜色映射，annot=False：不显示数值标签。
'''
print('相关系数矩阵')
a1=dataset.corr()
print(a1)
df = pd.DataFrame(a1)

# 保存为 Excel 文件
excel_file_path = '/Users/weiliangyu/github.push.project/Diabetes.xlsx'
df.to_excel(excel_file_path, index=False, sheet_name='Sheet1')
'''
data = read_csv('/Users/weiliangyu/Desktop/diabetes.csv')
X = data.iloc[:, :-1]#提取所有行和除最后一列外的所有列（特征矩阵）。
y = data.iloc[:, -1]#提取最后一列作为目标变量（标签，如糖尿病诊断结果）。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)#test_size=0.2：测试集占总数据的 20%。random_state=1：随机种子，确保结果可复现。
imputer = Imputer(missing_values=0,strategy='median')#missing_values=0：指定缺失值为 0（在某些数据集中，0 可能表示缺失值）。strategy='median'：使用中位数作为填充值。
X_train2 = imputer.fit_transform(X_train)#fit_transform()：在训练集上拟合 imputer 并转换训练集。计算训练集各特征的中位数，并替换训练集中的 0 值
X_test2 = imputer.transform(X_test)#transform()：使用训练集的统计信息（如中位数）转换测试集。使用训练集计算的中位数填充测试集的 0 值
X_train3 = pd.DataFrame(X_train2)#将numpy.ndarray类型的X_train2转回pandas.DataFrame类型。
y_train3 = pd.DataFrame(y_train)#将numpy.ndarray类型的X_train2转回pandas.DataFrame类型。
print(type(y_train),type(X_train3),y_train.shape[0],X_train3.shape)
result = pd.concat([X_train3, y_train3], axis=1)#将X_train3和y_train3按列（axis=1）拼接在一起，生成一个新的DataFrame result。
result.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']#为result的列命名。
print(result.columns)
plotHistogram(result,"Outcome","Insulin",'Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(result,"Outcome",'SkinThickness','SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
labels = {0:'Pregnancies',1:'Glucose',2:'BloodPressure',3:'SkinThickness',4:'Insulin',5:'BMI',6:'DiabetesPedigreeFunction',7:'Age'}
print(labels)
print("\nColumn #, # of Zero Values\n")
print((X_train3[:] == 0).sum())#计算X_train3中每一列中零值的数量。

compareABunchOfDifferentModelsAccuracy(X_train2, y_train)
defineModels()
'''
LR = 逻辑回归 
RF = 随机森林分类器 
KNN = K 近邻分类器 
SVM = 支持向量机分类器
LSVM = 线性支持向量分类器
GNB = 高斯朴素贝叶斯
DTC = 决策树分类器
GBC = 梯度提升分类器
'''
# iterate over classifiers; adapted from https://www.kaggle.com/hugues/basic-ml-best-of-10-classifiers
results = {}# 初始化一个空字典，用于存储分类器的名称和对应的交叉验证分数
for name, clf in zip(names, classifiers):# 遍历分类器名称和分类器列表
    scores = cross_val_score(clf, X_train2, y_train, cv=5)#cross_val_score：Scikit-learn 中的交叉验证函数。clf：当前分类器对象。cv=5：将训练数据分为 5 个子集（折），依次用 4 折训练、1 折验证。
    results[name] = scores# 将当前分类器的名称和对应的交叉验证分数存储到 results 字典中
for name, scores in results.items():# 遍历 results 字典中的每个分类器名称和对应的交叉验证分数
    print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100*scores.mean(), 100*scores.std() * 2))
def runDecisionTree(a, b, c, d):#a, b: 训练数据的特征和标签c, d: 测试数据的特征和标签
    model = DecisionTreeClassifier()#创建默认参数的决策树分类器，可能过拟合
    accuracy_scorer = make_scorer(accuracy_score)#创建准确率评分器，但后续使用字符串 'accuracy' 作为评分标准。
    model.fit(a, b)#用训练数据 (a, b) 拟合决策树模型。
    kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=7)#KFold(n_splits=10): 将训练集划分为 10 个子集
    accuracy = model_selection.cross_val_score(model, a, b, cv=kfold, scoring='accuracy')#cross_val_score: 返回 10 次交叉验证的准确率数组。
    mean = accuracy.mean() 
    stdev = accuracy.std()#计算交叉验证准确率的平均值和标准差
    prediction = model.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)#对测试集 c 进行预测，生成混淆矩阵。
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plot_learning_curve(model, 'Learning Curve For DecisionTreeClassifier', a, b, (0.60,1.1), 10)#plot_learning_curve: 绘制模型在不同训练样本量下的学习曲线，评估模型泛化能力
    #learning_curve(model, 'Learning Curve For DecisionTreeClassifier', a, b, (0.60,1.1), 10)
    plt.show()
    plot_confusion_matrix(cnf_matrix, classes=dict_characters,title='Confusion matrix')#可视化混淆矩阵，展示模型在各类别上的预测效果
    plt.show()
    print('DecisionTreeClassifier - Training set accuracy: %s (%s)' % (mean, stdev))#打印交叉验证的平均准确率和标准差
    return
runDecisionTree(X_train2, y_train, X_test2, y_test)
feature_names1 = X.columns.values#获取数据集的特征名称
def plot_decision_tree1(a,b):
    dot_data = tree.export_graphviz(a, out_file=None, #export_graphviz: 将决策树转换为 DOT 格式（图形描述语言）out_file=None 表示：不是写入文件，而是保存在变量中（字符串形式）
                             feature_names=b,  
                             class_names=['Healthy','Diabetes'],  #类标签的名字（0 表示 Healthy，1 表示 Diabetes），会显示在叶子节点
                             filled=False, rounded=True,  #filled=False：是否为每个节点填充颜色，False 表示不填色rounded=True：是否使用圆角边框，图形更美观
                             special_characters=False)  #special_characters=False：是否允许特殊字符，如 >、≤，为 False 则不使用
    graph = graphviz.Source(dot_data)  #graphviz.Source: 解析 DOT 数据并生成可视化图形
    return graph 
clf1 = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=12)#max_depth=3: 决策树的最大深度为 3（限制树的复杂度）。min_samples_leaf=12: 每个叶节点至少包含 12 个样本（防止过拟合）。
clf1.fit(X_train2, y_train)
tree1=plot_decision_tree1(clf1,feature_names1)# 调用 plot_decision_tree1 函数，传入决策树 clf1 和特征名称 feature_names1，绘制决策树。
tree1.render('/Users/weiliangyu/Desktop/decision_tree', format='png', view=True)
tree1.view()

feature_names = X.columns.values#feature_names：获取特征名称列表（用于后续特征重要性分析）
clf1 = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=12)#ax_depth=3：树的最大深度为 3 层（防止过拟合）。min_samples_leaf=12：每个叶节点至少包含 12 个样本（防止过拟合）
clf1.fit(X_train2, y_train)#.fit()：使用训练数据 X_train2 和标签 y_train 训练模型。
print('Accuracy of DecisionTreeClassifier: {:.2f}'.format(clf1.score(X_test2, y_test)))#.score()：在测试集上计算模型准确率。
columns = X.columns
coefficients = clf1.feature_importances_.reshape(X.columns.shape[0], 1)#feature_importances_：决策树自动计算的特征重要性分数（值越大越重要）。reshape(X.columns.shape[0], 1)：将特征重要性分数转换为与特征名称对应的形状。reshape()：将一维数组转为二维（匹配特征数量）
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)#pd.concat()：将特征名和重要性分数合并为 DataFrame，并按重要性降序排序。
print('DecisionTreeClassifier - Feature Importance:')
print('\n',fullList,'\n')

feature_names = X.columns.values
clf2 = RandomForestClassifier(max_depth=3,min_samples_leaf=12)
clf2.fit(X_train2, y_train)
print('Accuracy of RandomForestClassifier: {:.2f}'.format(clf2.score(X_test2, y_test)))#随机森林原理：集成多棵决策树，通过投票或平均提高稳定性和泛化能力。

columns = X.columns
coefficients = clf2.feature_importances_.reshape(X.columns.shape[0], 1)#clf2.feature_importances_ 获取随机森林集成后的特征重要性。
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('RandomForestClassifier - Feature Importance:')
print('\n',fullList,'\n')

clf3 = XGBClassifier()#XGBClassifier：梯度提升树模型（默认参数）优势：支持正则化、并行计算、处理缺失值，通常比传统决策树更高效。max_depth=6（树的深度，此处未显式设置）。learning_rate=0.3（学习率，控制每次迭代的步长）。n_estimators=100（树的数量）。
clf3.fit(X_train2, y_train)
print('Accuracy of XGBClassifier: {:.2f}'.format(clf3.score(X_test2, y_test)))
columns = X.columns
coefficients = clf3.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('XGBClassifier - Feature Importance:')
print('\n',fullList,'\n')


data = read_csv('/Users/weiliangyu/Desktop/diabetes.csv')
data2 = data.drop(['Pregnancies','BloodPressure','DiabetesPedigreeFunction', 'Age','SkinThickness','Insulin'], axis=1)#从数据集中删除 6 个特征，只保留一部分特征，用于构建一个降维模型。axis=1 表示按列删除。
X2 = data2.iloc[:, :-1]#取除了最后一列以外的所有列（也就是特征）。
y2 = data2.iloc[:, -1]#取最后一列（目标变量，即是否患糖尿病）。
X_train3, X_test3, y_train3, y_test3 = train_test_split(X2, y2, test_size=0.2, random_state=1)#test_size=0.2：20%为测试集，80%为训练集。random_state=1：确保每次划分一致，保证可重复性。
imputer = Imputer(missing_values=0,strategy='median')
X_train3 = imputer.fit_transform(X_train3)#在训练集上拟合填补规则，并将缺失值替换为中位数。返回的结果是一个 NumPy 数组。
X_test3 = imputer.transform(X_test3)#使用前面训练集上学习到的规则，填补测试集中的 0 值。
clf4 = XGBClassifier()#创建一个默认参数的 XGBoost 分类器。
clf4.fit(X_train3, y_train3)#用降维后的训练数据训练模型。
print('Accuracy of XGBClassifier in Reduced Feature Space: {:.2f}'.format(clf4.score(X_test3, y_test3)))#评估模型在测试集上的准确率，保留两位小数
columns = X2.columns
coefficients = clf4.feature_importances_.reshape(X2.columns.shape[0], 1)#clf4.feature_importances_：提取每个特征的重要性（贡献度），越高表示对结果影响越大。reshape：重新构造为列向量，方便后面组合。
absCoefficients = abs(coefficients)#因为特征重要性通常非负，所以这里abs是多余的，但保证安全。
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)#将变量名与其对应的重要性值结合成一个 DataFrame，并按重要性降序排列。
print('\nXGBClassifier - Feature Importance:')
print('\n',fullList,'\n')#打印出降维模型中各特征的重要性排名

clf3 = XGBClassifier()
clf3.fit(X_train2, y_train)
print('\n\nAccuracy of XGBClassifier in Full Feature Space: {:.2f}'.format(clf3.score(X_test2, y_test)))#打印基于全部特征模型的测试集准确率
columns = X.columns
coefficients = clf3.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('XGBClassifier - Feature Importance:')
print('\n',fullList,'\n')