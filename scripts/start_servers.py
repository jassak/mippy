#!/usr/bin/env python

import time
import subprocess
from pathlib import Path


mippy_path = Path.cwd().parent / "mippy"

server_names = ["serverA", "serverB", "serverC"]
# server_names = ["serverA", "serverB"]

print("Starting name server")
ns_proc = subprocess.Popen("pyro5-ns")

proc = {}

try:
    for name in server_names:
        proc[name] = subprocess.Popen(
            ["python", "server.py", "--servername={name}".format(name=name)],
            cwd=mippy_path.as_posix(),
        )
    time.sleep(1200)
    # while True:
    #     pass
except KeyboardInterrupt:
    pass
finally:
    print()
    for name in server_names:
        proc[name].kill()
        print(f"{name} killed")
    ns_proc.kill()
    print("name server killed")
