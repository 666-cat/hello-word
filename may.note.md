#may.18th
##git操作
git pull origin develop从develop--分支拉取到本地分支
git push origin develop--本地分支推送到develop远程分支
git push -u origin <分支名>--当前分支与远程分支建立关联，以后直接用git push就行
git branch --unset-upstream--此命令仅仅是切断了本地分支和远程分支之间的关联
git push--推送本地分支到远程追踪分支
git pull--拉取远程分支到本地追踪分支
git add 文件名--添加文件到缓冲区
git commit -m "简介"--提交所有缓冲区的项目
git checkout main--切换到本地已存在的main分支
git switch develop--切换到develop分支
git status--查看所有未追踪的文件
git add .--添加所有变更到缓冲区
git commit -m "所有的共同简介"
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
docker ps -a 查看当前正在运行的容器
操作	命令
查看容器状态	docker ps
停止容器	docker stop mysql-container
启动容器	docker start mysql-container
删除容器	docker rm mysql-container
查看日志	docker logs mysql-container
进入容器终端	docker exec -it mysql-container bash
备份数据	docker cp mysql-container:/var/lib/mysql /host/path/backup
