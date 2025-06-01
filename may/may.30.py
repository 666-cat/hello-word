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
data = pd.read_csv('/Users/weiliangyu/Downloads/Medicine_Details.csv')

# 创建游标对象
cursor = mydb.cursor()

# 插入数据
create_table_query="""
    CREATE TABLE IF NOT EXISTS Medicine_Details (
    id INT AUTO_INCREMENT PRIMARY KEY,
    `Medicine Name` VARCHAR(255) NOT NULL COMMENT '药品名称',
    Composition TEXT COMMENT '成分',
    Uses TEXT COMMENT '用途',
    Side_effects TEXT COMMENT '副作用',
    `Image URL` VARCHAR(512) COMMENT '图片链接',
    Manufacturer VARCHAR(255) COMMENT '制造商',
    `Excellent Review %` DECIMAL(5, 2) COMMENT '好评率',
    `Average Review %` DECIMAL(5, 2) COMMENT '中评率',
    `Poor Review %` DECIMAL(5, 2) COMMENT '差评率'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='药品信息表';
    """
cursor.execute(create_table_query)
print("学生信息表创建成功")
sheet = data
insert_query = """
    INSERT INTO Medicine_Details (`Medicine Name`,Composition,Uses,Side_effects,`Image URL`,Manufacturer,`Excellent Review %`,`Average Review %`,`Poor Review %`)
    VALUES ( %s, %s,%s,%s,%s,%s,%s,%s,%s)
    """
    # 逐行读取数据（跳过表头）
i=1
for _, row in data[2:].iterrows():
    data_tuple = (
        row['Medicine Name'], row['Composition'], row['Uses'],
        row['Side_effects'], row['Image URL'], row['Manufacturer'],
        row['Excellent Review %'], row['Average Review %'], row['Poor Review %']
    )
    cursor.execute(insert_query, data_tuple)
    mydb.commit()
    i=i+1

    print(f"成功插入 {i} 条记录")
    print(f"成功插入 {cursor.rowcount} 条记录")

# 提交更改
mydb.commit()

print(cursor.rowcount, "记录插入成功。")