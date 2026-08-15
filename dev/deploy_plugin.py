"""Push a workdir plugin into the running container and hot-reload it.

Usage:
    python dev/deploy_plugin.py selfie [strava ...]

Why this exists: the manual loop is three fiddly steps, each with a trap.

1. `docker cp` fails here with "copying between containers is not supported",
   so the file is streamed in with `tee` instead.
2. The plugin must live in `plugins/workdir/` -- `collector.py` only scans
   that directory -- and its key must be listed in PLUGIN_ENABLED.
3. Reloading beats restarting: a container restart resets the memory
   consolidation clock (MEMORY_CONSOLIDATION_MIN_INTERVAL_SECONDS), which
   makes it look like memory is broken for the next five minutes.
"""

import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Same reason as the utf-8 in wsl(), but for our own output: the log lines we
# echo are Chinese, and a stock Windows console encodes stdout as cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
DISTRO = "NekroAgent"
CONTAINER = "nekro_agent"
DEST_DIR = "/root/nekro_agent_data/plugins/workdir"
BASE = "http://127.0.0.1:8021"
LAUNCHER_CFG = Path.home() / "AppData/Local/NekroAgent/config.json"


def wsl(cmd: str) -> subprocess.CompletedProcess:
    # utf-8 is not optional: the container logs in Chinese and Windows would
    # otherwise decode them as cp1252 and raise.
    return subprocess.run(
        ["wsl", "-d", DISTRO, "--", "sh", "-c", cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def to_wsl_path(p: Path) -> str:
    """C:\\AI\\x -> /mnt/c/AI/x"""
    s = p.resolve().as_posix()
    return f"/mnt/{s[0].lower()}{s[2:]}"


def admin_token() -> str:
    with open(LAUNCHER_CFG, encoding="utf-8") as f:
        launcher = json.load(f)
    pw = (launcher.get("deploy_info") or {}).get("admin_password")
    if not pw:
        sys.exit(f"no deploy_info.admin_password in {LAUNCHER_CFG}")

    req = urllib.request.Request(
        BASE + "/api/user/login",
        data=json.dumps({"username": "admin", "password": pw}).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read().decode())
    token = (res.get("data") or {}).get("access_token") or res.get("access_token")
    if not token:
        sys.exit(f"login succeeded but no token in response: {json.dumps(res)[:200]}")
    return token


def reload(module: str, token: str) -> None:
    req = urllib.request.Request(
        f"{BASE}/api/plugins/reload?module_name={urllib.parse.quote(module)}",
        data=b"{}",
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            print(f"  reload: {r.status} {r.read().decode()[:120]}")
    except urllib.error.HTTPError as e:
        sys.exit(f"  reload failed: {e.code} {e.read().decode()[:300]}")


def main() -> None:
    modules = sys.argv[1:]
    if not modules:
        sys.exit(__doc__)

    token = admin_token()

    for module in modules:
        src = REPO / "plugins" / "workdir" / f"{module}.py"
        if not src.exists():
            sys.exit(f"not found: {src}")

        print(f"{module}:")
        res = wsl(f"docker exec -i {CONTAINER} tee {DEST_DIR}/{module}.py < '{to_wsl_path(src)}' > /dev/null")
        if res.returncode != 0:
            sys.exit(f"  copy failed: {res.stderr.strip()[:300]}")
        print(f"  copied -> {DEST_DIR}/{module}.py")

        reload(module, token)

    # Surface load errors, which the reload endpoint reports as 200 regardless.
    logs = wsl(f"docker logs {CONTAINER} --tail 40")
    for line in (logs.stdout or "").splitlines() + (logs.stderr or "").splitlines():
        if any(s in line for s in ("插件加载成功", "插件加载失败", "Traceback", "ERROR")):
            print(f"  | {line.strip()}")


if __name__ == "__main__":
    main()
