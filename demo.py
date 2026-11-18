import argparse
import json
from contextlib import closing
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

from audit.decision import report, report_robust
from measure.catalog import load_revision
from measure.rationals import format_rational
from measure.revisions import register_batch
from publish.commit import PublicationError
from publish.store import initialize_archive, open_archive
from review.handoff import read_result
from train.freeze import FitInputError, fit


SAMPLES = Path(__file__).resolve().parent / 'samples'


def _load_sample(name):
    return json.loads((SAMPLES / name).read_text(encoding='utf-8'))


def _register(archive, batch, *, parent=None):
    return register_batch(archive, batch['name'], batch['unit'], batch['columns'], batch['rows'],
                          parent_rev=parent)


def _check(actual, expected, label):
    if actual != expected:
        raise AssertionError(f'{label}: expected {expected!r}, got {actual!r}')


def _snapshot(archive, fit_id, report_ids):
    frozen = read_result(archive, 'fit', fit_id)
    reports = [read_result(archive, 'report', report_id) for report_id in report_ids]
    return frozen, reports


def _report_counts(archive):
    return archive.execute('SELECT (SELECT count(*) FROM reports), (SELECT count(*) FROM report_links)').fetchone()


def _release_examples(archive, data, expected, revisions):
    training, left, right = (revisions[slot] for slot in ('training', 'left', 'right'))
    frozen_id = fit(archive, training, data['largest'])
    try:
        fit(archive, training, data['invalid_largest'])
    except FitInputError:
        _check(archive.execute('SELECT count(*) FROM fits').fetchone()[0], 1, 'invalid fit count')
    else:
        raise AssertionError('inaccurate highest-value declaration was accepted')

    published = [report(archive, frozen_id, left, right, *pair) for pair in data['thresholds']]
    report_ids = [report_id for report_id, _ in published]
    verdicts = [passed for _, passed in published]
    _check(verdicts, expected['coverage'], 'common-axis verdicts')
    snapshot = _snapshot(archive, frozen_id, report_ids)
    fit_view, report_views = snapshot
    _check({key: fit_view[key] for key in expected['fit']}, expected['fit'], 'frozen fit')
    _check(fit_view['train_rev'], training, 'fit source')
    for view, (minimum_left, minimum_right), passed in zip(
            report_views, data['thresholds'], expected['coverage'], strict=True):
        _check([view['fit_id'], view['train_rev'], view['left_rev'], view['right_rev']],
               [frozen_id, training, left, right], 'report sources')
        _check([view['lo_left'], view['lo_right'], view['passed']],
               [minimum_left, minimum_right, passed], 'readable report payload')
    return frozen_id, report_ids, verdicts, snapshot


def _interrupt_report(archive, frozen_id, revisions, thresholds):
    before = _report_counts(archive)
    staged = []

    def interrupt():
        staged.append(_report_counts(archive))
        raise RuntimeError('simulated interruption before commit')

    try:
        report(archive, frozen_id, revisions['left'], revisions['right'], *thresholds,
               before_commit=interrupt)
    except PublicationError:
        _check(staged, [tuple(count + 1 for count in before)], 'staged report and link')
        _check(_report_counts(archive), before, 'report rollback')
    else:
        raise AssertionError('interrupted publication succeeded')
    return before


def _read_corrected_sources(archive, old_revisions, new_revisions):
    sources = {slot: load_revision(archive, revision) for slot, revision in new_revisions.items()}
    for slot, source in sources.items():
        _check(source.parent_rev, old_revisions[slot], f'{slot} correction parent')
    return {slot: [format_rational(value) for value in source.rows[0]]
            for slot, source in sources.items()}


def _robust_examples(archive, frozen_id, robust):
    """Judge the robust sample reports against the frozen XY plane."""
    batches = {slot: _register(archive, batch) for slot, batch in robust['batches'].items()}
    cases = []
    for specification in robust['reports']:
        report_id, passed = report_robust(
            archive, frozen_id, batches[specification['left']], batches[specification['right']],
            specification['lo_left'], specification['lo_right'],
            specification['drop_left'], specification['drop_right'],
        )
        _check(passed, specification['passed'], f"robust verdict {specification}")
        view = read_result(archive, 'report', report_id)
        _check([view['drop_left'], view['drop_right']],
               [specification['drop_left'], specification['drop_right']], 'readback budgets')
        cases.append((specification, report_id, view))

    single_ids = []
    for slot, rows in robust['singles'].items():
        revision = register_batch(archive, slot, 'mm/s', ('X', 'Y', 'Z'), rows)
        single_ids.append(revision)
    pairwise = [report(archive, frozen_id, first, second, '1/2', '1/2')[1]
                for first, second in combinations(single_ids, 2)]
    _check(pairwise, [True, True, True], 'each pair shares an axis')
    joint_left = register_batch(archive, 'three directions', 'mm/s', ('X', 'Y', 'Z'),
                                [row for rows in robust['singles'].values() for row in rows])
    joint = report_robust(archive, frozen_id, joint_left, batches['quiet'], '1/2', 0, 2, 0)[1]
    _check(joint, False, 'all three together share no axis')
    return batches, cases, pairwise


def _robust_axis_archive(archive, data):
    """The single-axis sample: the same revision is used for both positions."""
    training = _register(archive, data['batches']['axis_training'])
    frozen_id = fit(archive, training, data['axis_largest'])
    shared = _register(archive, data['batches']['shared'])

    report_ids = []
    for specification in data['axis_reports']:
        report_id, passed = report_robust(
            archive, frozen_id, shared, shared, specification['lo_left'], specification['lo_right'],
            specification['drop_left'], specification['drop_right'])
        _check(passed, specification['passed'], f"axis verdict {specification}")
        report_ids.append(report_id)
    corrections = _register(archive, {'name': '共用读数', 'unit': 'mm/s',
                                      'columns': ('X', 'Y', 'Z'),
                                      'rows': data['correction']['rows']}, parent=shared)
    return frozen_id, shared, report_ids, corrections


def _emit(case_id, inputs, expected, actual, downstream_expected, downstream_actual, paths, oracle):
    evidence = {
        'case_id': case_id,
        'input': inputs,
        'expected': expected,
        'actual': actual,
        'downstream_expected': downstream_expected,
        'downstream_actual': downstream_actual,
        'code_paths': paths,
        'oracle': oracle,
    }
    print('POINTWISE_SCENARIO ' + json.dumps(evidence, ensure_ascii=False, sort_keys=True))


def run_demo():
    data = _load_sample('day.json')
    expected = _load_sample('expected.json')
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'demo.sqlite3'
        with closing(open_archive(path)) as archive:
            initialize_archive(archive)
            revisions = {slot: _register(archive, batch) for slot, batch in data['batches'].items()}
            frozen_id, report_ids, verdicts, snapshot = _release_examples(archive, data, expected, revisions)
            counts = _interrupt_report(archive, frozen_id, revisions, data['thresholds'][0])
            corrected = {
                slot: _register(archive, data['batches'][slot] | changes, parent=revisions[slot])
                for slot, changes in data['corrections'].items()
            }
            _, robust_cases, pairwise = _robust_examples(archive, frozen_id, data['robust'])
            axis_fit, axis_shared, axis_ids, axis_correction = _robust_axis_archive(archive, data['robust'])
            axis_snapshot = [read_result(archive, 'report', identifier) for identifier in axis_ids]
            end_counts = _report_counts(archive)
        for _ in range(2):
            with closing(open_archive(path)) as archive:
                initialize_archive(archive)
                _check(_report_counts(archive), end_counts, 'durable publications and rollback')
                reread = _snapshot(archive, frozen_id, report_ids)
                _check(reread, snapshot, 'fixed handoff after reopening')
                corrected_first_rows = _read_corrected_sources(archive, revisions, corrected)
                _check(corrected_first_rows, expected['corrected_first_rows'], 'durable corrections')
                axis_reread = [read_result(archive, 'report', identifier) for identifier in axis_ids]
                _check(axis_reread, axis_snapshot, 'robust fixed handoff after reopening')
                _check(load_revision(archive, axis_correction).parent_rev, axis_shared,
                       'robust correction parent')
        with closing(open_archive(path)) as archive:
            initialize_archive(archive)
            retry_id, passed = report(archive, frozen_id, revisions['left'], revisions['right'],
                                      *data['thresholds'][0])
            _check(passed, expected['coverage'][0], 'retry verdict')
            _check(read_result(archive, 'report', retry_id)['passed'], passed, 'durable retry')

    frozen, report_views = reread
    _emit('v6.translated_cases', {'batches': data['batches'], 'largest': data['largest'],
                                  'thresholds': data['thresholds']},
          expected['coverage'], verdicts, expected['coverage'], [view['passed'] for view in report_views],
          ['train/freeze.py', 'audit/energy.py', 'audit/criterion.py', 'audit/decision.py',
           'review/report_view.py'],
          'The two unit-axis energies add to one. A diagonal reaches both halves exactly.')
    original_sources = [revisions[slot] for slot in ('training', 'left', 'right')]
    _emit('v9.archive_reopen', {'source_ids': original_sources, 'corrections': data['corrections']},
          original_sources, [frozen['train_rev'], report_views[0]['left_rev'], report_views[0]['right_rev']],
          expected['corrected_first_rows'], corrected_first_rows,
          ['measure/catalog.py', 'publish/store.py', 'publish/links.py', 'review/handoff.py'],
          'Compare the published parents with the IDs originally passed in and the new rows with the inputs.')

    robust_expected = [specification['passed'] for specification in data['robust']['reports']]
    _emit('v10.robust_retained_budget',
          {'reports': data['robust']['reports'],
           'pairwise_single_rows': data['robust']['pairwise']},
          robust_expected, [view['passed'] for _, _, view in robust_cases],
          [True, True, True], pairwise,
          ['audit/robust.py', 'audit/request.py', 'audit/decision.py', 'publish/report_records.py',
           'review/report_view.py'],
          'Each pair of the three single-row conditions shares an axis, yet all three together share none; '
          'the worst retained mean stays exact and each report keeps its per-position budgets.')
    _emit('v11.robust_budget_reopen',
          {'fit_id': axis_fit, 'revision': axis_shared,
           'budgets': [[specification['drop_left'], specification['drop_right']]
                       for specification in data['robust']['axis_reports']],
           'correction': axis_correction},
          [specification['passed'] for specification in data['robust']['axis_reports']],
          [view['passed'] for view in axis_snapshot],
          [[view['drop_left'], view['drop_right']] for view in axis_snapshot],
          [[view['drop_left'], view['drop_right']] for view in axis_reread],
          ['audit/robust.py', 'publish/report_records.py', 'publish/links.py', 'measure/catalog.py',
           'review/handoff.py'],
          'Two reports of the same revision keep their own budgets; a correction is a new parent, '
          'and reopening reads the fixed reports without recomputing.')


def main():
    parser = argparse.ArgumentParser(description='Three-axis vibration review archive')
    commands = parser.add_subparsers(dest='command', required=True)
    initialize = commands.add_parser('init', help='initialize a local archive')
    initialize.add_argument('archive', type=Path)
    commands.add_parser('demo', help='run the sample day in an isolated archive')
    options = parser.parse_args()
    if options.command == 'init':
        with closing(open_archive(options.archive)) as archive:
            initialize_archive(archive)
    else:
        run_demo()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
