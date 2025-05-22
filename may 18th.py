"""
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
"""
"""
#python装饰器
def decorator_function(original_function):
    def wrapper(*args,**kwargs):
        before_call_code()
        result =original_function(*args,**kwargs)
        after_call_code()
        return result
    return wrapper
#使用装饰器
@decorator_function
def target_function(arg1,arg2):
    pass
"""
"""
@time_logger
def target_function():
    pass
等同于
def target function():
    pass
target_function=time_longer(target_function)
"""
"""
#实例
def my_decorator(func):
    def wrapper():
        print("before")
        func()
        print("after")
    return wrapper
@my_decorator
def say_hello():
    print("Hello!")
say_hello()
"""
#实例
"""
def my_decorator(func):
    def wrapper(*arg,**kwarg):
        print("before")
        func(*arg,**kwarg)
        print("after")
    return wrapper
@my_decorator
def greet(name):
    print(f"hello, {name}")
greet("Alice")
"""
"""
def repeat(num_times):
    def decorator(func):
        def wrapper():
            for _ in range(num_times):
                func()
        return wrapper
    return decorator
@repeat(3)
def say_hello():
    print("hello")
say_hello()
"""
"""
def log_class(cls):
    class wrapper:
        def __init__(self,*arg,**kwarg):
            self.wrapped = cls(*arg,**kwarg)
        def _getattr_(self,name):
            return getattr(self.wrapped,name)
        def display(self):
            print(f"调用{cls.__name__}.display() 前")
            self.wrapped.display()
            print(f"调用{cls.__name__}.display() 后")
    return wrapper
@log_class
class myclass:
    def display(self):
        print("这是myclass的display方法")
obj=myclass()
obj.display()
"""
"""
class singletondecorator:
    def __init__(self,cls):
        self.cls =cls
        self.instance=None
    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance=self.cls(*args,**kwargs)
        return self.instance
@singletondecorator
class database:
    def __init__(self):
        print("database 初始化")
db1=database()
db2=database()
print(db1 is db2)
"""
"""
class myclass:
    @staticmethod
    def static_method():
        print("this is a static method")
    @classmethod
    def class_method(cls):
        print(f"this is a class_method of{cls.__name__}.")
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value
myclass.static_method()
myclass.class_method()
obj=myclass()
obj.name = "Alice"
print(obj.name)
"""
"""
def decorator1(func):
    def wrapper():
        print("decorator 1")
        func()
    return wrapper
def decorator2(func):
    def wrapper():
        print("decorator 2")
        func()
    return wrapper
@decorator1
@decorator2
def say_hello():
    print("hello")
say_hello()
"""