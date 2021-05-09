dev:
	./scripts/start_env.sh docker/dev

dev-build:
	./scripts/start_env.sh docker/dev --build
	
attach:
	docker-compose exec ${MODE} bash
