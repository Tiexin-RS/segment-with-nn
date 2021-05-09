#! /bin/bash
echo Linking $1 env into parent directory...
rm -rf Dockerfile* docker-compose.yml
ln -s $1/docker-compose.yml docker-compose.yml
ln -s $1/Dockerfile Dockerfile
ln -s $1/Dockerfile.tvm Dockerfile.tvm

echo Starting containers...
# echo DEBUG: pwd is `pwd`
shift
docker-compose up -d $@
