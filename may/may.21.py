#模块
"""
import sys
print("该文件路径如下")
for i in sys.argv:#sys.argv包含脚本名,以及后续参数
    print(i)
print(sys.path)#查找模块对应的文件所在的路径
"""
"""
#support.py代码
def print_fun(per):
    print("hello:",per)
    return

import support
support.print_fun("bob")
>>>hello:bob
"""
'''
#这个模块命名为fibo.ty
def fib(n):
    a,b=0,1
    while b<n:
        print(b,end=" ")
        a,b=b,b+a
def fib1(n):
    a,b=0,1
    c=[]
    while b<n:
        c.append(b)
        a,b=b,a+b
    print(c)
fib1(100)
fib(100)
'''
'''
import fibo
fibo.fib(1000)就可以直接运行了
也可以
from fibo import fib
fib(1000)
from fibo import fib as nb
nb(1000)#都是一样的
'''
'''
if __name__=="__main__":
    print("程序在自身运行")#每一个模块都有一个name,直接运行该模块name的名字为main,被导入name为模块名
else:
    print("程序在另外一个模块进行")
import sys
a=dir(sys)#dir()函数可以找到模块内定义的所有名称，以字符串列表的形式返回
print(a)
c=5
b=dir()#没有参数就会得到当前模块所定义的属性列表
print(b)
'''
'''
s='hello bob'
print(str(s))
print(repr(s))
print(type(repr(s)))
print(str(1/7))
print(repr(1/7))
x=100*3.5
y=10*0.35
s='x的值为'+repr(x),'y的值为'+str(y)
print(s)
#repr()函数可以转义字符串里面的特殊字符
hello='hello\nbob'
hellos=repr(hello)
print(hello)
print(hellos)
'''
for i in range(1,11):
    print(repr(i).rjust(2),repr(i**2).rjust(2),repr(i**3).rjust(4,'0'))
    s=repr(i).rjust(2),repr(i**2).rjust(2),repr(i**3).rjust(4,'0')
    print(s,end=" ")#使用end就会在输出结果后面加空格，下一个输出的结果就不用换行了
#str.format()的基本用法：括号及里面的字符会被format后面的参数替换
print('{}:{}'.format(123,456))
print('站点列表{0:.3f},{1}和{2}'.format(100.12345,'2','3'))
print('站点列表 {0}, {1}, 和 {other}。'.format('Google', 'Runoob', other='Taobao'))
print('站点列表 {1}, {0}, 和 {other}。'.format('Google', 'Runoob', other='Taobao'))
table={'google':1,'runoob':2,'taobao':3}
for name,number in table.items():
    print('{0:10}==>{1:10}'.format(name,number))#:后面接一个整数，可以保证该域的宽度
table = {'Google': 1, 'Runoob': 2, 'Taobao': 3}
print('Runoob: {0[Runoob]:d}; Google: {0[Google]:d}; Taobao: {0[Taobao]:d}'.format(table))#0 代表的是 format() 方法传递的第 1 个参数的索引
print('Runoob: {Runoob:d}; Google: {Google:d}; Taobao: {Taobao:d}'.format(**table))
str = input("请输入：");
print ("你输入的内容是: ", str)
f=open("/Users/weiliangyu/Desktop/21统计88人.xlsx", "r")
print(f)
