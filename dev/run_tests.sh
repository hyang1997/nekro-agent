#!/usr/bin/env bash
# 在容器里跑测试。
#
# 宿主机上没有装 nekro 的依赖，容器的生产镜像里又没有 pytest，所以用一个在生产镜像上
# 加装 pytest 的派生镜像（nekro-test），把宿主机源码挂进去跑——不碰正在运行的 nekro_agent 容器。
#
# 首次使用先建镜像：
#   wsl -d NekroAgent -- sh -c 'printf "FROM kromiose/nekro-agent:latest\nRUN /app/.venv/bin/pip install --no-cache-dir pytest pytest-asyncio pytest-mock\n" > /tmp/test.dockerfile && docker build -t nekro-test:latest -f /tmp/test.dockerfile /tmp'
#
# 用法：
#   bash dev/run_tests.sh                          # 全量
#   bash dev/run_tests.sh tests/test_foo.py -q     # 透传给 pytest
set -euo pipefail

SRC_IN_WSL="/mnt/c/AI/nekro-agent"
ARGS="${*:-tests/}"

exec wsl -d NekroAgent -- sh -c "docker run --rm \
  -v ${SRC_IN_WSL}:/src -w /src \
  -e PYTHONPATH=/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  nekro-test:latest /app/.venv/bin/python3 -m pytest ${ARGS}"
