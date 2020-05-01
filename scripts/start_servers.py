import subprocess
from pathlib import Path
import time
from mippy.baseclasses import node_names

mippy_path = Path.cwd().parent / "mippy"


print("Starting name server and local servers\n")
ns_proc = subprocess.Popen("pyro5-ns")

proc = {}
for name in node_names:
    proc[name] = subprocess.Popen(
        ["python", "localnode.py", "--servername={name}".format(name=name)],
        cwd=mippy_path.as_posix(),
    )
    time.sleep(0.5)

time.sleep(2)
print()
input("Press enter to kill processes")
for name in node_names:
    proc[name].kill()
ns_proc.kill()
