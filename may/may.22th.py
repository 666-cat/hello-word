import mysql.connector
from mysql.connector import Error
import random
import pandas as pd  # 用于查看结果，可选择性安装

# 数据库配置
db_config = {
    'host': 'localhost',
    'user': 'root',  # 替换为你的MySQL用户名
    'password': 'qazwsx',  # 替换为你的MySQL密码
    'database': 'app_db',  # 数据库名称
    "auth_plugin": "mysql_native_password",
    'raise_on_warnings': True
}

# 随机生成姓名
def generate_name(gender):
    # 常见姓氏
    last_names = ['赵', '钱', '孙', '李', '周', '吴', '郑', '王', '陈', '杨', '黄', '赵', '吴', '周']
    
    # 名字（按性别区分）
    male_first_names = ['伟', '强', '磊', '洋', '勇', '军', '杰', '浩', '峰', '超', '明', '刚', '平', '辉']
    female_first_names = ['芳', '娜', '秀英', '敏', '静', '雅', '丽', '强', '磊', '洋', '勇', '军', '杰', '浩']
    
    last_name = random.choice(last_names)
    
    if gender == '男':
        first_name = random.choice(male_first_names)
    else:
        first_name = random.choice(female_first_names)
    
    return f"{last_name}{first_name}"

# 随机生成性别
def generate_gender():
    return random.choice(['男', '女'])

# 随机生成出生年份（1950-2010之间）
def generate_birth_year():
    return random.randint(1950, 2010)

# 随机生成出生月份
def generate_birth_month():
    return random.randint(1, 12)

# 创建数据库和表
def create_database():
    try:
        # 连接到MySQL服务器
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        # 创建表（如果不存在）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS person_info (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(20) NOT NULL,
            gender ENUM('男', '女') NOT NULL,
            birth_year INT NOT NULL,
            birth_month INT NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')
        
        connection.commit()
        print("数据库和表创建成功")
        
    except Error as e:
        print(f"数据库错误: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# 插入随机数据
def insert_random_data(count=100):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        for _ in range(count):
            gender = generate_gender()
            name = generate_name(gender)
            birth_year = generate_birth_year()
            birth_month = generate_birth_month()
            
            cursor.execute(
                "INSERT INTO person_info (name, gender, birth_year, birth_month) VALUES (%s, %s, %s, %s)",
                (name, gender, birth_year, birth_month)
            )
        
        connection.commit()
        print(f"成功插入 {count} 条数据")
        
    except Error as e:
        print(f"数据库错误: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# 查看生成的数据（可选）
def view_data():
    try:
        connection = mysql.connector.connect(**db_config)
        df = pd.read_sql_query("SELECT * FROM person_info", connection)
        connection.close()
        
        print("\n数据预览:")
        print(df.head())
        print(f"\n总记录数: {len(df)}")
    except ImportError:
        print("\n需要安装pandas库才能查看数据: pip install pandas")
    except Error as e:
        print(f"数据库错误: {e}")

if __name__ == "__main__":
    # 创建数据库和表
    create_database()
    
    # 插入100条随机数据
    insert_random_data(100)
    
    # 查看数据（可选）
    view_data()
import requests
base_url="http://localhost:3000/api"
def get_tocken():
    login_data={
        "username":"wei1hu@icloud.com",
        "password":"wly1314521"
    }
    response=requests.post(f"{base_url}/session",json=login_data)
    if response.status_code==200:
        return response.json()
    else:
        raise Exception(f"获取数据列表失败:{response.text}")
tocken=get_tocken()
print(f"获得的Token:{tocken}")