# TVM 环境简易使用文档

## 实用命令
```bash
cd <ProjectDir>
# Option1: just start all services
make dev
# Option2: rebuild all image and start
make dev-build

# attach to tvm environment
docker-compose exec tvm bash
```

## 注意
- 通常本人会把 `docker-compose` alias 成 `dc` 方便实用
- `/opt/segelectri` 下有完整的项目的代码，`/workspace` 下有 tvm 的项目代码
