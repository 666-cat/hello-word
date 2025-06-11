docker run -d \
  --name mysql-june \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=qazwsx \
  -e MYSQL_DATABASE=wlydatabase \
  -e MYSQL_USER=wly \
  -e MYSQL_PASSWORD=qazwsx \
  -v mysql-data:/var/lib/mysql \
  mysql:8.0