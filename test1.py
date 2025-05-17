import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path

# 获取当前目录（兼容脚本和交互式环境）
try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

# 构建数据文件路径
file_path = current_dir / "数据.xlsx"

# 检查文件是否存在
if not file_path.exists():
    raise FileNotFoundError(f"文件不存在: {file_path}")

# 读取文件
df = pd.read_excel(file_path, sheet_name='Sheet1')
print(f"成功读取文件: {file_path}")

# 数据基本信息
print("\n数据基本信息:")
df.info()

# # 处理潜在异常值（以工业废水排放为例）
# q1 = df['工业废水排放总量'].quantile(0.25)
# q3 = df['工业废水排放总量'].quantile(0.75)
# iqr = q3 - q1
# df_clean = df[(df['工业废水排放总量'] <= q3 + 1.5*iqr)]

# # 重命名列（去除空格）
# df_clean.columns = df_clean.columns.str.strip()


# pd.set_option('display.max_rows', 500)  # 最多显示500行
# pd.set_option('display.max_columns', 50)  # 最多显示50列

# # 打印清洗后的数据
# print("\n数据全部内容信息:")
# print(df_clean.to_csv(sep='\t', na_rep='nan'))