#! /bin/bash
echo Linking $1 env into parent directory...
rm -rf Dockerfile docker-compose.yml
ln -s $1/docker-compose.yml docker-compose.yml
ln -s $1/Dockerfile Dockerfile

echo Starting containers...
# echo DEBUG: pwd is `pwd`
docker-compose up --build -d
