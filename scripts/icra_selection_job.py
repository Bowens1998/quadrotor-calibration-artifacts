"""Deterministic Slurm-array dispatch; commands and task index are logged."""
import argparse
import json
import os
import subprocess
import sys

from _common import ROOT


def command_for(phase, index):
    regimes = ['mass_1p4', 'lag_3']
    if phase == 'environment':
        assert index in [0, 1]
        return ['scripts/icra_selection_environment.py', '--split', ['validation', 'test'][index],
                '--device', 'cuda', '--batch-size', '3', '--protocol-lock',
                'runs/icra_selection_validation_20260912/LOCK.json']
    rows = []
    if phase == 'validation':
        rows = [('branches', r, p, None) for r in regimes for p in range(3)]
        rows += [('flight', r, p, m) for r in regimes for p in range(3) for m in range(8)]
    elif phase == 'test':
        rows = [('flight', r, 0, m) for r in regimes for m in range(10)]
    elif phase == 'pilot':
        rows = [('flight', r, 0, m) for r in regimes for m in [0, 3, 9]]
    else:
        raise ValueError(phase)
    operation, regime, panel, model = rows[index]
    command = ['scripts/icra_selection_run.py', operation, '--split', phase,
               '--regime', regime, '--panel', str(panel), '--device', 'cuda']
    if model is not None:
        command += ['--index', str(model)]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['pilot', 'environment', 'validation', 'test'])
    parser.add_argument('--index', required=True, type=int)
    args = parser.parse_args()
    command = [sys.executable, '-u', *command_for(args.phase, args.index)]
    print(json.dumps(dict(phase=args.phase, index=args.index, command=command,
                          slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                          host=os.environ.get('HOSTNAME'))), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
