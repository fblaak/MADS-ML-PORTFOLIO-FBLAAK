"""Run a small pilot Ray Tune search for fast validation.

This script calls `hypertune.run_search` with a small budget and
saves a short summary to `pilot_result.txt` in the same folder.
"""
from pathlib import Path
from hypertune import run_search


def main():
    folder = Path(__file__).resolve().parent
    # small pilot
    analysis = run_search(num_samples=5, max_epochs=2, storage_dir=folder / 'logs' / 'ray_pilot')

    try:
        best = analysis.get_best_config(metric='test_loss', mode='min')
    except Exception:
        best = None

    out = folder / 'pilot_result.txt'
    with out.open('w') as f:
        f.write('pilot best config:\n')
        f.write(str(best) + '\n')

    print('Pilot finished. Results written to', out)


if __name__ == '__main__':
    main()
