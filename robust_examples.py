import argparse
import json
import unittest
from contextlib import closing
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

from audit.decision import report
from audit.request import CoverageInputError
from measure.revisions import register_batch
from publish.commit import PublicationError
from publish.store import initialize_archive, open_archive
from review.handoff import read_result
from train.freeze import fit


AXES = ('X', 'Y', 'Z')
UNIT = 'mm/s'
X, Y, ZERO = (1, 0, 0), (0, 1, 0), (0, 0, 0)
F = Fraction
TRIPLE = (X, (F(3, 5), F(4, 5), 0), (F(3, 5), -F(4, 5), 0))
ENERGY = 'worst_retained_mean_squared_projection_about_training_center'


def _register(archive, name, rows, *, parent_rev=None):
    return register_batch(archive, name, UNIT, AXES, rows, parent_rev=parent_rev)


def _fit(archive, *directions, center=ZERO):
    rows = [tuple(c + sign * value for c, value in zip(center, direction, strict=True))
            for direction in directions for sign in (1, -1)]
    revision = _register(archive, 'training', rows)
    return revision, fit(archive, revision, F(1, len(directions)))


def _check(actual, expected, label):
    if actual != expected:
        raise AssertionError(f'{label}: expected {expected!r}, got {actual!r}')


def baseline():
    with closing(open_archive(':memory:')) as archive:
        initialize_archive(archive)
        _, plane = _fit(archive, X, Y)
        _, axis = _fit(archive, X)
        quiet = _register(archive, 'quiet', [ZERO])
        first = _register(archive, 'with quiet row', [X, ZERO])
        second = _register(archive, 'without quiet row', [(F(3, 5), 0, 0), (F(4, 5), 0, 0)])
        full_means = [report(archive, axis, rev, quiet, '1/2', 0)[1] for rev in (first, second)]
        _check(full_means, [True, True], 'V1 equal full means of 1/2')
        singletons = [_register(archive, f'condition-{i}', [row]) for i, row in enumerate(TRIPLE)]
        pair_verdicts = [report(archive, plane, a, b, '1/2', '1/2')[1]
                         for a, b in combinations(singletons, 2)]
        _check(pair_verdicts, [True, True, True], 'V1 two-condition checks')
        return {'v1_equal_full_mean_verdicts': full_means, 'v1_pair_verdicts': pair_verdicts}


def _saved_view(archive, robust, args, expected, label):
    report_id, passed = robust(archive, *args)
    _check(passed is expected, True, label)
    frozen, left, right, lo_left, lo_right, drop_left, drop_right = args
    view = read_result(archive, 'report', report_id)
    training = read_result(archive, 'fit', frozen)
    thresholds = [f'{q.numerator}/{q.denominator}' for q in map(F, (lo_left, lo_right))]
    keys = ('fit_id', 'train_rev', 'left_rev', 'right_rev', 'lo_left', 'lo_right',
            'drop_left', 'drop_right', 'energy')
    _check([view[key] for key in keys],
           [frozen, training['train_rev'], left, right, *thresholds, drop_left, drop_right, ENERGY],
           f'{label} saved request')
    _check(view['passed'] is expected, True, f'{label} saved verdict')
    _check(view['report_id'], report_id, f'{label} report identity')
    return view


def _math_cases(archive, robust):
    _, plane = _fit(archive, X, Y)
    quiet = _register(archive, 'quiet', [ZERO])
    alternating = _register(archive, 'alternating axes', [X, Y])
    triple = _register(archive, 'three directions', TRIPLE)
    repeated = _register(archive, 'repeat contributes twice', [X, X, ZERO])
    x = _register(archive, 'repeat X', [X, X])
    y = _register(archive, 'repeat Y', [Y, Y])
    p = tuple(F(n, 3) for n in (2, 1, -2))
    q = tuple(F(n, 3) for n in (2, -2, 1))
    center = (2, -3, F(1, 2))
    _, oblique = _fit(archive, p, q, center=center)
    left = _register(archive, 'oblique P', [tuple(a + b for a, b in zip(center, p, strict=True))] * 2)
    right = _register(archive, 'oblique Q', [tuple(a + b for a, b in zip(center, q, strict=True))] * 2)
    zero_budget = report(archive, plane, x, y, '1/3', '2/3')[1]
    cases = (
        ('closed_order_tie', (plane, alternating, quiet, '1/2', 0, 1, 0), True),
        ('each_retained_set_has_different_axis', (plane, alternating, quiet, '3/4', 0, 1, 0), False),
        ('pairwise_only', (plane, triple, quiet, '1/2', 0, 2, 0), False),
        ('duplicates_and_retained_denominator', (plane, repeated, quiet, '1/2', 0, 1, 0), True),
        ('irrational_axes', (plane, x, y, '1/3', '2/3', 1, 1), True),
        ('raise_right_minimum', (plane, x, y, '1/3', '7/10', 1, 1), False),
        ('vertical_axis', (plane, x, y, 0, 1, 1, 1), True),
        ('oblique_metric_and_center', (oblique, left, right, '1/3', '2/3', 1, 1), True),
        ('zero_budgets', (plane, x, y, '1/3', '2/3', 0, 0), zero_budget),
    )
    return {label: _saved_view(archive, robust, args, expected, label)['passed']
            for label, args, expected in cases}


def _stored_rows(archive):
    return tuple(line for line in archive.iterdump() if line.startswith('INSERT INTO '))


def _withheld(archive, action, expected_error):
    before = _stored_rows(archive)
    try:
        action()
    except expected_error:
        _check(_stored_rows(archive), before, 'archive rows are unchanged')
    else:
        raise AssertionError(f'expected {expected_error.__name__}')


def _publication_cases(archive, robust):
    _, axis = _fit(archive, X)
    rev = _register(archive, 'shared condition', [X, ZERO])
    previous_id, previous_passed = report(archive, axis, rev, rev, '1/2', 0)
    _check(previous_passed is True, True, 'V1 whole-batch verdict')
    previous = read_result(archive, 'report', previous_id)
    fixed = (axis, rev, rev, 0, 0)
    saved = [_saved_view(archive, robust, (*fixed, a, b), True, 'ordered budgets')
             for a, b in ((1, 0), (0, 1))]
    _check(saved[0]['report_id'] != saved[1]['report_id'], True, 'distinguishable reports')
    for drops in ((True, 0), (0, 2), (-1, 0), (0, '1')):
        _withheld(archive, lambda: robust(archive, *fixed, *drops), CoverageInputError)
    oversized = _register(archive, 'oversized condition', [X] * 13)
    _withheld(archive, lambda: robust(archive, axis, rev, oversized, 0, 0, 0, 0), CoverageInputError)

    before = _stored_rows(archive)
    staged = []

    def interrupt():
        staged.append(_stored_rows(archive))
        raise RuntimeError('interrupt publication before commit')

    args = (axis, rev, rev, '1/2', 0, 1, 0)
    _withheld(archive, lambda: robust(archive, *args, before_commit=interrupt), PublicationError)
    _check(len(staged), 1, 'before_commit was called once')
    _check(staged[0] != before, True, 'publication had staged rows')
    saved.append(_saved_view(archive, robust, args, False, 'retry after rollback'))
    _register(archive, 'shared condition', [X, X], parent_rev=rev)
    return [previous, *saved]


def _run_regression():
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).resolve().parent / 'tests'),
                            top_level_dir=str(Path(__file__).resolve().parent))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('original regression suite failed')
    return result.testsRun


def verify():
    from audit.decision import report_robust

    regression_count = _run_regression()
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'audit.sqlite3'
        with closing(open_archive(path)) as archive:
            initialize_archive(archive)
            verdicts = _math_cases(archive, report_robust)
            saved = _publication_cases(archive, report_robust)
        with closing(open_archive(path)) as archive:
            initialize_archive(archive)
            reloaded = [read_result(archive, 'report', view['report_id']) for view in saved]
        _check(reloaded, saved, 'read fixed reports after correction and reopen')
        return {'regression_tests': regression_count, 'coverage': verdicts,
                'after_reopen': reloaded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'verify'))
    args = parser.parse_args()
    results = baseline() if args.mode == 'baseline' else verify()
    print(json.dumps(results, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
