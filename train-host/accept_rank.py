"""Apply the frozen skill acceptance limits to baseline and candidate reports.

The report has no power to change a candidate or thresholds. Request regression
is evaluated separately on recorded needs when fresh rows have no such needs.
Usage: accept_rank.py BASELINE CANDIDATE OUT
"""
import json
import sys
from pathlib import Path


def assess(baseline, candidate):
    a, b = baseline['skills'], candidate['skills']
    checks = {}
    for band in ['rare', 'mid', 'common']:
        old, new = a['bands'][band], b['bands'][band]
        supported = old['support'] == new['support'] and old['support'] > 0
        margin = .05 if band in ['rare', 'mid'] else -.02
        checks[band + '_pooled'] = supported and new['visible@6'] >= old['visible@6'] + margin
        if band in ['rare', 'mid']:
            checks[band + '_macro'] = supported and old['macro_support3'] is not None and new['macro_support3'] > old['macro_support3']
    regressions = []
    for name, old in a['per_skill'].items():
        new = b['per_skill'].get(name)
        if new is None or new['support'] != old['support']:
            regressions.append(name)
        elif old['support'] >= 5 and old['visible@6'] / old['support'] >= .5 and new['visible@6'] / new['support'] < .5:
            regressions.append(name)
    checks['no_category_collapse'] = not regressions
    checks['preload_precision'] = b['preload_precision'] is not None and b['preload_precision'] >= .9
    checks['negative_false_preloads'] = b['no_skill_rows'] > 0 and b['no_skill_preload_rate'] <= .05
    checks['preload_coverage'] = b.get('relevant_preloaded', 0) >= .8 * a.get('relevant_preloaded', 0)
    return {'checks': checks, 'category_regressions': regressions, 'skills_passed': all(checks.values()),
            'note': 'Request regression, runtime parity, and application smoke checks are separate gates.'}


def main():
    baseline, candidate, output = sys.argv[1:]
    result = assess(json.loads(Path(baseline).read_text()), json.loads(Path(candidate).read_text()))
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['skills_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
