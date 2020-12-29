dev:
	./scripts/start_env.sh docker/dev
	
attach:
	docker-compose exec ${MODE} bash