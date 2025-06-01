# may.18th
## git操作
git pull origin develop从develop--分支拉取到本地分支
git push origin develop--本地分支推送到develop远程分支
git push -u origin <分支名>--当前分支与远程分支建立关联，以后直接用git push就行
git branch --unset-upstream--此命令仅仅是切断了本地分支和远程分支之间的关联
git push--推送本地分支到远程追踪分支
git pull--拉取远程分支到本地追踪分支
git add 文件名--添加文件到缓冲区，也就是追踪
git commit -m "简介"--提交所有缓冲区的项目添加到本地仓库
git ls-remote origin--查看远程仓库的分支和文件
git checkout main--切换到本地已存在的main分支
git switch develop--切换到develop分支
git status--查看所有未追踪的文件
git add .--添加所有变更到缓冲区
git commit -m "所有的共同简介"添加到本地仓库
##docker创建容器
'''
docker run -d \
  --name mysql-container \          容器名称
  -p 3306:3306 \                   端口映射（宿主机:容器）
  -e MYSQL_ROOT_PASSWORD=your_root_password \   root 用户密码
  -e MYSQL_DATABASE=your_database \           初始化数据库（可选）
  -e MYSQL_USER=your_user \                    创建普通用户（可选）
  -e MYSQL_PASSWORD=your_user_password \       普通用户密码（可选）
  -v mysql-data:/var/lib/mysql \               数据卷持久化数据
  mysql:8.0
'''
docker里面有我所建立的容器，容器里有我所建立的数据库
docker ps -a 查看当前正在运行的容器
操作	命令
查看容器状态	docker ps
停止容器	docker stop mysql-container
启动容器	docker start mysql-container
删除容器	docker rm mysql-container
查看日志	docker logs mysql-container
进入容器终端	docker exec -it mysql-container bash
登陆信息 mysql -u root -p
查看容器ip  docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mysql-prod
备份数据	docker cp mysql-container:/var/lib/mysql /host/path/backup
端口容器映射  docker port mysql-prod
import mysql.connector:数据库连接模块
from openpyxl import load_workbook:Excel处理模块
cursor=conn.cursor():创建游标对象，用于执行 SQL 语句并获取结果
cursor.execute(要执行的sql语句)
cursor.fetchone()获取下一行数据，返回单个元组；结果集为空时返回 None。 核心逻辑是移动结果集指针并返回当前行数据，其行为类似于迭代器。
commit():每次插入数据都要执行一次
row函数可以逐行读取数据，row是一个元组
cursor.fetchall():获取所有剩余行的数据，返回元组列表 [(row1), (row2), ...]；结果集为空时返回空列表 []
pip freeze > requirements.txt 更新 requirements.txt



