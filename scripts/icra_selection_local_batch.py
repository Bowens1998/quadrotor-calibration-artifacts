"""Local dispatch of the fixed experimental job mapping."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import os
import subprocess
import sys
import time

from icra_selection_job import command_for
from icra_selection_run import BASE, ROOT, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['validation', 'test'])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    assert 1 <= args.workers <= 2
    logdir = BASE / 'logs' / ('local_' + args.phase)
    logdir.mkdir(parents=True, exist_ok=False)
    n = 54 if args.phase == 'validation' else 20
    def run(index):
        command = [sys.executable, '-u', *command_for(args.phase, index)]
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        start = time.monotonic()
        with (logdir / f'job{index:02d}.log').open('x') as stream:
            stream.write(json.dumps(dict(command=command, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())) + '\n')
            stream.flush()
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = dict(index=index, exit_code=result.returncode, wall_s=time.monotonic()-start,
                      command=command, log_sha256=sha(logdir / f'job{index:02d}.log'))
        write(logdir / f'job{index:02d}.json', record)
        print('JOB_FINISHED', args.phase, index, result.returncode, flush=True)
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = [future.result() for future in as_completed([executor.submit(run, i) for i in range(n)])]
    write(logdir / 'dispatch.json', dict(phase=args.phase, workers=args.workers,
          dispatcher_source_sha256=sha(__file__), locked_mapping_source_sha256=sha(ROOT / 'scripts/icra_selection_job.py'),
          results=sorted(results, key=lambda row: row['index'])))
    assert all(row['exit_code'] == 0 for row in results), 'Failed jobs retained; inspect exact logs before any retry'


if __name__ == '__main__':
    main()
