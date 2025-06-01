#!/usr/bin/python3
'''
#类定义
class people:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))
 
#单继承示例
class student(people):
    grade = ''
    def __init__(self,n,a,w,g):
        #调用父类的构函
        people.__init__(self,n,a,w)
        self.grade = g
    #覆写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
 
#另一个类，多继承之前的准备
class speaker():
    topic = ''
    name = ''
    def __init__(self,n,t):
        self.name = n
        self.topic = t
    def speak(self):
        print("我叫 %s，我是一个演说家，我演讲的主题是 %s"%(self.name,self.topic))
 
#多继承
class sample(speaker,student):
    a =''
    def __init__(self,n,a,w,g,t):
        student.__init__(self,n,a,w,g)
        speaker.__init__(self,n,t)
        print(n,a,w,g,t)
 
test = sample("Tim",25,80,4,"Python")
print(test)   #方法名同，默认调用的是在括号中参数位置排前父类的方法
'''
"""
git pull origin develop
git push origin develop
git push -u origin develop
git branch --unset-upstream develop
git push
git pull
git add wenjianming
git commit -m ""
git ls-remote origin
git checkout main
git switch develop
git status
git add .
git commit -m ""
docker ps -a
docker ps
docker stop 容器id
docker start 容器id
docker rm 容器id
docker logs 容器id
docker exec -it 容器id bash
mysql -u root -p
docker inspect -f'{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 容器id
docker cp mysql-container:/var/lib/mysql /home/ubuntu/mysql-data
docker port 容器id
import mysql connector
from openpyxl import load_workbook
cursor=conncursor()
cursor.execute("SELECT * FROM students")
cursor.fechall()
cursor.fechone()
commit()
pip freeze > requirements.txt
"""