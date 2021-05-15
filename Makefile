dev:
	./scripts/start_env.sh docker/dev up -d

dev-build:
	./scripts/start_env.sh docker/dev up -d --build

attach:
	./scripts/start_env.sh docker/dev exec ${POD_NAME} bash
	
