import pandas as pd  # 导入 pandas 库，用于数据处理和分析
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于数据可视化
import plotly.express as px
import seaborn as sns  # 导入 seaborn 库，基于 matplotlib 的高级数据可视化库
from sklearn.preprocessing import LabelEncoder  # 从 sklearn 库中导入 LabelEncoder 类，用于将分类变量编码为数值变量

med = pd.read_csv('/Users/weiliangyu/Downloads/Medicine_Details.csv')  # 从指定路径读取 CSV 文件并存储到 med 变量中
med.drop_duplicates(inplace=True)  # 删除 med 数据框中的重复行，直接在原数据框上进行修改

# 为每列创建独立的 LabelEncoder 实例
encoder_medicine = LabelEncoder()  # 创建一个 LabelEncoder 实例，用于编码药品名称
encoder_manufacturer = LabelEncoder()  # 创建一个 LabelEncoder 实例，用于编码制造商名称
med['Medicine Name'] = encoder_medicine.fit_transform(med['Medicine Name'])  # 将药品名称列转换为数值编码并更新到原数据框中
med['Manufacturer'] = encoder_manufacturer.fit_transform(med['Manufacturer'])  # 将制造商名称列转换为数值编码并更新到原数据框中

print(med.head(6))  # 打印 med 数据框的前 6 行
med.info()  # 查看 med 数据框的基本信息，包括列名、数据类型、非空值数量等
print('重复行数:', med.duplicated().sum())  # 打印 med 数据框中重复行的数量
print('每列缺失值数量:\n', med.isnull().sum())  # 打印 med 数据框每列的缺失值数量

# 设置 seaborn 的绘图风格为 'darkgrid'
sns.set_style('darkgrid')

# 创建包含 3 个子图的画布
fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # 创建一个 1 行 3 列的子图布局，画布大小为 15x5

# 绘制 Excellent Review % 的直方图
sns.histplot(med['Excellent Review %'], kde=True, ax=ax[0], color='green')  # 绘制直方图并显示核密度估计曲线，指定子图位置和颜色
ax[0].set_title('Distribution of Excellent Review %')  # 设置第一个子图的标题

# 绘制 Average Review % 的直方图
sns.histplot(med['Average Review %'], kde=True, ax=ax[1], color='blue')  # 绘制直方图并显示核密度估计曲线，指定子图位置和颜色
ax[1].set_title('Distribution of Average Review %')  # 设置第二个子图的标题

# 绘制 Poor Review % 的直方图
sns.histplot(med['Poor Review %'], kde=True, ax=ax[2], color='red')  # 绘制直方图并显示核密度估计曲线，指定子图位置和颜色
ax[2].set_title('Distribution of Poor Review %')  # 设置第三个子图的标题

plt.tight_layout()  # 自动调整子图之间的间距，避免重叠
plt.show()  # 显示绘制好的图表

manufacturer_counts = med['Manufacturer'].value_counts()  # 统计每个制造商的药品数量
print(manufacturer_counts.head(10))  # 打印数量最多的前 10 个制造商的统计结果

import plotly.express as px  # 导入 plotly 的 express 模块，用于创建交互式图表

# 绘制每个制造商的药品数量柱状图
fig = px.bar(manufacturer_counts, x=manufacturer_counts.index, y=manufacturer_counts.values, title='Number Of Medicines by Manufacturer')  # 创建柱状图
fig.show()  # 显示绘制好的图表

fig.update_xaxes(tickangle=45)  # 旋转 x 轴标签 45 度，提高可读性
fig.update_layout(xaxis_title='Manufacturer', yaxis_title='Number of Medicines')  # 设置 x 轴和 y 轴的标题
 # 打印图表对象，显示图表的 HTML 代码
from mpl_toolkits.mplot3d import Axes3D  # 从 mpl_toolkits.mplot3d 模块导入 Axes3D 类，用于创建 3D 图表

# 假设 Composition 是分类列，使用 LabelEncoder 进行编码
encoder_composition = LabelEncoder()  # 创建一个 LabelEncoder 实例，用于编码药品成分
med['Composition_encoded'] = encoder_composition.fit_transform(med['Composition'])  # 将药品成分列转换为数值编码并更新到原数据框中

fig = plt.figure(figsize=(12, 12))  # 创建一个大小为 12x12 的画布
ax = fig.add_subplot(111, projection='3d')  # 在画布上添加一个 3D 子图

# 使用编码后的 Composition 列作为颜色映射
sc = ax.scatter(med['Excellent Review %'], med['Average Review %'], med['Poor Review %'], c=med['Composition_encoded'], marker='o')  # 创建 3D 散点图

ax.set_xlim([0, 100])  # 设置 x 轴的显示范围为 0 到 100
ax.set_ylim([0, 100])  # 设置 y 轴的显示范围为 0 到 100
ax.set_zlim([0, 100])  # 设置 z 轴的显示范围为 0 到 100

ax.set_xlabel('Excellent Review %')  # 设置 x 轴的标签
ax.set_ylabel('Average Review %')  # 设置 y 轴的标签
ax.set_zlabel('Poor Review %', labelpad=20)  # 设置 z 轴的标签，并调整标签与轴的间距

plt.colorbar(sc, label='Composition')  # 添加颜色条，用于显示药品成分编码的映射关系

print(ax.zaxis.label)  # 打印 z 轴的标签

plt.title('3D Scatter Plot of Reviews with Composition Encoding')  # 设置 3D 散点图的标题
plt.show()  # 显示绘制好的 3D 散点图