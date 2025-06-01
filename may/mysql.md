-- 创建数据库，如果数据库不存在则创建
CREATE DATABASE IF NOT EXISTS database_name;
-- 用法：创建一个新的数据库，IF NOT EXISTS 用于避免数据库已存在时出错

-- 使用指定数据库
USE database_name;
-- 用法：切换当前操作的数据库为指定的数据库

-- 删除数据库
DROP DATABASE IF EXISTS database_name;
-- 用法：删除指定的数据库，IF EXISTS 用于避免数据库不存在时出错-- 创建表
CREATE TABLE IF NOT EXISTS table_name (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INT,
    email VARCHAR(255) UNIQUE
);
-- 用法：创建一个新的表，IF NOT EXISTS 避免表已存在时出错。定义了表的列名、数据类型和约束

-- 查看表结构
DESCRIBE table_name;
-- 用法：显示指定表的列信息，包括列名、数据类型、是否允许为空等

-- 修改表结构，添加列
ALTER TABLE table_name ADD COLUMN new_column VARCHAR(255);
-- 用法：在指定表中添加一个新的列

-- 修改表结构，删除列
ALTER TABLE table_name DROP COLUMN new_column;
-- 用法：从指定表中删除一个列

-- 删除表
DROP TABLE IF EXISTS table_name;
-- 用法：删除指定的表，IF EXISTS 避免表不存在时出错-- 插入数据
INSERT INTO table_name (name, age, email) VALUES ('John Doe', 30, 'john@example.com');
-- 用法：向指定表中插入一行数据，指定列名和对应的值

-- 查询数据
SELECT * FROM table_name;
-- 用法：查询指定表中的所有行和列

SELECT name, age FROM table_name WHERE age > 25;
-- 用法：查询指定表中年龄大于 25 的行的姓名和年龄列

-- 更新数据
UPDATE table_name SET age = 31 WHERE name = 'John Doe';
-- 用法：将指定表中姓名为 'John Doe' 的行的年龄更新为 31

-- 删除数据
DELETE FROM table_name WHERE name = 'John Doe';
-- 用法：删除指定表中姓名为 'John Doe' 的行-- 排序查询结果
SELECT * FROM table_name ORDER BY age DESC;
-- 用法：查询指定表中的所有行，并按年龄降序排序

-- 分组查询
SELECT department, AVG(salary) FROM employees GROUP BY department;
-- 用法：按部门分组，计算每个部门的平均工资