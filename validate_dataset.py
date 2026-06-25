#!/usr/bin/env python3
import subprocess, sys
subprocess.run([sys.executable, "run_hgai.py", "--lite", "--validate"], check=False)
