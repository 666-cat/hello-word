import pandas as pd
import mysql.connector

# 连接到 MySQL 数据库
mydb = mysql.connector.connect(
    **{"host":"localhost",
    "port":3306,
    "user":"root",
    "password":"qazwsx",
    "database":"app_db",
    "auth_plugin": "mysql_native_password"}
)

# 读取 CSV 文件
data = pd.read_csv('/Users/weiliangyu/Downloads/diabetes.csv')

# 创建游标对象
cursor = mydb.cursor()

# 插入数据
create_table_query="""
    CREATE TABLE IF NOT EXISTS Diabetes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Pregnancies INT COMMENT '怀孕次数',
    Glucose INT COMMENT '葡萄糖浓度',
    BloodPressure INT COMMENT '血压mmHg',
    SkinThickness INT COMMENT '皮肤厚度mm',
    Insulin INT COMMENT '胰岛素水平mu U/ml',
    BMI DECIMAL(5, 2) COMMENT '身体质量指数kg/m²',
    DiabetesPedigreeFunction DECIMAL(5, 3) COMMENT '糖尿病遗传函数值',
    Age INT COMMENT '年龄（岁）',
    Outcome INT COMMENT '结果0=未患糖尿病1=患糖尿病）'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='糖尿病相关指标表';
    """
cursor.execute(create_table_query)
print("学生信息表创建成功")
sheet = data
insert_query = """
    INSERT INTO Diabetes (
    Pregnancies, Glucose, BloodPressure, SkinThickness, 
    Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    # 逐行读取数据（跳过表头）
i = 1
for _, row in data[2:].iterrows():#_：忽略行索引（因为你可能不关心具体是第几行）
    # 将数据转换为 Python 原生类型
    data_tuple = (
        int(row['Pregnancies']),
        int(row['Glucose']),
        int(row['BloodPressure']),
        int(row['SkinThickness']),
        int(row['Insulin']),
        float(row['BMI']),
        float(row['DiabetesPedigreeFunction']),
        int(row['Age']),
        int(row['Outcome'])
    )
    cursor.execute(insert_query, data_tuple)
    mydb.commit()
    i += 1
    print(f"成功插入 {i} 条记录")
    print(f"成功插入 {cursor.rowcount} 条记录")

# 提交更改
mydb.commit()

print(cursor.rowcount, "记录插入成功。")