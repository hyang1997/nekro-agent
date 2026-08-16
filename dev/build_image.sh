#!/bin/sh
# Build the NekroAgent image from this fork and point the launcher at it.
#
# Run inside the WSL distro (building from /mnt/c is painfully slow):
#   wsl -d NekroAgent -- sh -c "cd /root/nekro-agent-src && sh dev/build_image.sh"
#
# Two things this exists to stop you forgetting:
#
# 1. TZ. dockerfile defaults to Asia/Shanghai to match upstream. The agent
#    talks about the time and schedules its own reminders in container-local
#    time, so a wrong zone quietly poisons everything downstream -- it will
#    cheerfully tell you it set a 9am reminder in the wrong 9am.
# 2. The :latest tag. docker-compose.yml (owned by the Windows launcher)
#    pulls kromiose/nekro-agent:latest, so a build that isn't retagged is a
#    build the launcher ignores.

set -e

TZ_ARG="${TZ_ARG:-America/Toronto}"
SHA="$(git rev-parse --short HEAD)"
TAG="kromiose/nekro-agent:hao-${SHA}"

echo "building ${TAG}  (TZ=${TZ_ARG})"
docker build --build-arg "TZ=${TZ_ARG}" -t "${TAG}" .

# The launcher watches :latest. Upstream builds stay reachable under their own
# tags, so retagging here loses nothing.
docker tag "${TAG}" kromiose/nekro-agent:latest
echo "tagged ${TAG} -> kromiose/nekro-agent:latest"

echo
echo "next:  cd /root/nekro_agent && docker compose up -d nekro_agent"
echo "check: docker exec nekro_agent date"
