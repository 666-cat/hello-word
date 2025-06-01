import pandas as pd
from pathlib import Path
try:
    current_dir=Path(__file__).parent
except NameError:
    current_dir=Path.cwd()
file_path=current_dir/"数据.xlsx"
if not file_path.exists():
    raise FileNotFoundError(f"文件不存在:{file_path}")
df=pd.read_excel(file_path,sheet_name='Sheet1')
print(f"read successfully",{file_path})
print("数据基本信息")
df.info()
print("数据全部内容:")
print(df.to_csv(sep="\t",na_rep="nan"))
print("相关系数矩阵：")
a1=df.corr()
print(a1)