"""One-command entry point for the reproducible current audit package."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/locked_rule_grid.yaml'); p.add_argument('--out',default='outputs/locked/grsl_current_audit'); args=p.parse_args()
 # The wrapper deliberately rebuilds only evidence that is available locally.
 subprocess.run([sys.executable,'reproduce.py','--out',args.out],cwd=ROOT,check=True)
 print('status: reference audit rebuilt; full rule-grid and lockbox remain explicitly pending')
if __name__=='__main__': main()
