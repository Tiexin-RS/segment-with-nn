#! /bin/bash
DOCKER_DIR=$1
shift
OPS=$1
shift

if [[ ${OPS} == "up" ]]; then
	echo Linking ${DOCKER_DIR} env into parent directory...
	rm -rf Dockerfile* docker-compose.yml
	ln -s ${DOCKER_DIR}/docker-compose.yml docker-compose.yml
	ln -s ${DOCKER_DIR}/Dockerfile Dockerfile
	ln -s ${DOCKER_DIR}/Dockerfile.tvm Dockerfile.tvm
fi

echo Starting containers...
# echo DEBUG: pwd is `pwd`
if [[ ${DC_PREFIX} ]]; then
	docker-compose -p ${DC_PREFIX} ${OPS} $@
else
	docker-compose ${OPS} $@
fi
