"""Pack the selected HelixSet TheraPep Val/Test reports, excluding weights/caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import io
import json
import sys


def main():
    root, archive = Path(sys.argv[1]), Path(sys.argv[2])
    tasks, suites = sys.argv[3].split(','), sys.argv[4].split(',')
    metrics = ('accuracy', 'precision', 'coverage', 'absolute_true', 'absolute_false',
               'micro_f1', 'macro_f1', 'micro_ap', 'macro_ap')
    rows, paths = [], []
    for task in tasks:
        for suite in suites:
            directory = root/task/suite
            result = json.loads((directory/'results.json').read_text(encoding='utf-8'))
            for split in ('val', 'test'):
                report = json.loads((directory/f'{split}.json').read_text(encoding='utf-8'))
                for scope, values in (('all_outputs', report['metrics']),
                                      ('positive_functions48', report['therapep']['positive_functions48'])):
                    rows.append(dict(task=task, suite=suite, split=split, scope=scope,
                        rows=report['rows'] if scope == 'all_outputs' else report['groups']['therapeutic_positive']['rows'],
                        best_epoch=result['best_epoch'], threshold=json.dumps(report['threshold']),
                        **{k: values[k] for k in metrics}))
            paths.extend(p for p in directory.iterdir()
                         if p.is_file() and p.suffix in ('.json', '.jsonl', '.npz', '.log'))
            log = root/f'{task}_{suite}.log'
            if log.is_file():
                paths.append(log)
        queue_log = root/f'{task}.log'
        if queue_log.is_file():
            paths.append(queue_log)
        evaluation_log = root/f'{task}_evaluate.log'
        if evaluation_log.is_file():
            paths.append(evaluation_log)
    with ZipFile(archive, 'x', ZIP_DEFLATED) as out:
        for path in paths:
            out.write(path, path.relative_to(root).as_posix())
        summary = io.StringIO(newline='')
        writer = csv.DictWriter(summary, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        out.writestr('summary.csv', summary.getvalue().encode('utf-8-sig'))
    print('RESULT ZIP:', archive.resolve())


if __name__ == '__main__':
    main()
