"""
a=[66.25,333,333,1,1234.5]
print(a.count(333),a.count(66.25),a.count("x"))
a.insert(2,1)
a.append(333)
print(a)
a.index(333)
print(a)
a.remove(333)
print(a)
a.reverse()
print(a)
a.sort()
print(a)
print(a.copy())
"""
"""
列表当栈用
stack=[]
stack.append(1)
stack.append(2)
stack.append(3)
print(stack)
top_element=stack.pop()
print(top_element)
print(stack)
top_element=stack[-1]
print(top_element)
is_empty=len(stack)==0
print(is_empty)
size=len(stack)
print(size)
"""
"""
class stack:
    def __init__(self):
        self.stack=[]
    def push(self,item):
        self.stack.append(item)
    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            raise IndentationError("pop form empty stack")
    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            raise IndentationError("peek from empty stack")
    def is_empty(self):
        return len(self.stack)==0
    def size(self):
        return len(self.stack)
stack=stack()
stack.push(1)
stack.push(2)
stack.push(3)
print("栈顶元素：",stack.peek())
print("栈大小",stack.size())
print("弹出元素",stack.pop())
print("栈是否为空",stack.is_empty())
print("栈大小",stack.size())
"""
"""
from collections import deque
queue=deque()
queue.append("a")
queue.append("b")
queue.append("c")
print("队列状态",queue)
first_element=queue.popleft()
print("移出的元素",first_element)
print("队列状态",queue)
first_element=queue[0]
print("队首元素",first_element)
is_empty=len(queue)==0
print("是空队列吗",is_empty)
size=len(queue)
print("队列大小",size)
"""
"""
列表表示队列
queue=[]
queue.append("a")
queue.append("b")
queue.append("c")
print("队列状态",queue)
first_element=queue.pop(0)
print("被移除的元素",first_element)
print("当前队列",queue)
check_first=queue[0]
print("当前队列第一个元素",check_first)
is_empty=len(queue)==0
print("队列是否为空",is_empty)
size=len(queue)
print("队列大小",size)
"""
"""
class queue:
    def __init__(self):
        self.queue=[]
    def enqueue(self,num):
        self.queue.append(num)
    def is_empty(self):
        return len(self.queue)==0
    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        else:
            raise IndentationError("dequeue from empty queue")
    def peek(self):
        if not self.is_empty==0:
            return self.queue[0]
        else:
            raise IndentationError("peek from empty queue")
    def size(self):
        return len(self.queue)
queue=queue()
queue.enqueue("a")
queue.enqueue("b")
queue.enqueue("c")
print("队列全体元素",queue)
print("队列首元素",queue.peek())
print("移出的队列首元素",queue.dequeue())
print("队列大小",queue.size())
print("是否为空队列",queue.is_empty())
"""
'''
vec=[3,6,9]
a=[3*i for i in vec]
print(a)
b=[[i,3*i] for i in vec]
print(b)
freshfruit=["  banana","   good"," nice   "]
c=[weapon.strip() for weapon in freshfruit]
print(c)
d=[x*3 for x in vec if x>=6]
print(d)
e=[x*3 for x in vec if x<6]
print(e)'''
'''
vec1=[2,4,6]
vec2=[3,6,9]
a=[x*y for x in vec1 for y in vec2]
b=[x+y for x in vec1 for y in vec2]
c=[vec1[i]*vec2[i] for i in range(len(vec1))]
print(a)
print(b)
print(c)
'''
import mysql.connector

# 连接参数（需替换为实际配置）
config = {
    "host": "localhost",       # 宿主机 IP
    "port": 3307,              # 宿主机映射的端口
    "user": "root",            # 容器内数据库的用户名
    "password": "your_password",  # 容器内数据库的密码
    "database": "test_db"      # 数据库名（若容器内未创建需先创建）
}

try:
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    cursor.execute("SELECT DATABASE();")
    result = cursor.fetchone()
    print(f"已连接到数据库：{result[0]}")
    
except mysql.connector.Error as err:
    print(f"连接失败：{err}")
    
finally:
    if conn.is_connected():
        cursor.close()
        conn.close()