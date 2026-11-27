"""Run each interrupt experiment in a fresh, time-limited process."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


probe = Path(sys.argv[1]).resolve()
fixtures = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="dstan-interrupt-") as directory:
    source = Path(directory) / "interrupt_probe.stan"
    shutil.copyfile(fixtures / source.name, source)
    shutil.copyfile(fixtures / "user_header.hpp", source.parent / "user_header.hpp")
    subprocess.run([probe, "--compile-interrupt", source], check=True)
    library = source.with_name("interrupt_probe_model.so")
    launch = {}
    if os.name == "nt":
        # Ctrl+Break is delivered to the child's private console.
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        launch = dict(creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
    else:
        launch = dict(start_new_session=True)
    for workers in (1, 2):
        result = subprocess.run([probe, "--interrupt", library, str(workers), *sys.argv[2:]],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=30, **launch)
        print(result.stdout, end="", flush=True)
        result.check_returncode()
