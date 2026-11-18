from fractions import Fraction
from itertools import combinations

from audit.decision import report, report_robust
from audit.request import CoverageInputError
from publish.commit import PublicationError
from review.handoff import read_result
from tests.support import ArchiveCase, OFFSET, X, Y, ZERO, shifted, training_rows
from train.freeze import fit


THRESHOLD = (1, 0)


class RobustCoverageTests(ArchiveCase):
    def _plane(self, center=ZERO):
        source = self.register('training', training_rows(center, X, Y))
        return source, fit(self.archive, source, '1/2')

    def test_pairwise_feasible_rows_need_not_share_one_axis(self):
        plane = (Fraction(3, 5), Fraction(4, 5), 0)
        twin = (Fraction(3, 5), Fraction(-4, 5), 0)
        _, fit_id = self._plane()
        triple = self.register('three directions', [X, plane, twin])
        quiet = self.register('quiet', [ZERO])
        report_id, passed = report_robust(self.archive, fit_id, triple, quiet,
                                          '1/2', 0, 2, 0)
        self.assertIs(passed, False)
        view = read_result(self.archive, 'report', report_id)
        self.assertEqual(view['energy'],
                         'worst_retained_mean_squared_projection_about_training_center')
        self.assertEqual((view['drop_left'], view['drop_right']), (2, 0))
        singles = [self.register(f'single {index}', [row])
                   for index, row in enumerate((X, plane, twin))]
        self.assertEqual([report(self.archive, fit_id, first, second, '1/2', '1/2')[1]
                          for first, second in combinations(singles, 2)],
                         [True, True, True])

    def test_budget_controls_the_retained_denominator(self):
        source = self.register('training', training_rows(ZERO, X))
        fit_id = fit(self.archive, source, 1)
        both = self.register('shared', [X, ZERO])
        first_id, first = report_robust(self.archive, fit_id, both, both, 0, 0, 1, 0)
        second_id, second = report_robust(self.archive, fit_id, both, both, 0, 0, 0, 1)
        self.assertIs(first, True)
        self.assertIs(second, True)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(read_result(self.archive, 'report', first_id)['drop_left'], 1)
        self.assertEqual(read_result(self.archive, 'report', second_id)['drop_left'], 0)

    def test_zero_budgets_reproduce_the_v1_verdict(self):
        for center in (ZERO, OFFSET):
            with self.subTest(center=center):
                source = self.register('training', training_rows(center, X, Y))
                fit_id = fit(self.archive, source, '1/2')
                left = self.register('left', [shifted(center, X)])
                right = self.register('right', [shifted(center, Y)])
                for thresholds in (('1/2', '1/2'), ('3/4', '3/4'), ('1/3', '2/3')):
                    whole = report(self.archive, fit_id, left, right, *thresholds)[1]
                    robust = report_robust(self.archive, fit_id, left, right, *thresholds, 0, 0)[1]
                    self.assertIs(robust, whole)

    def test_irrational_and_isolated_axes_survive_exact_judging(self):
        source = self.register('training', training_rows(ZERO, X, Y))
        fit_id = fit(self.archive, source, '1/2')
        repeat_x = self.register('repeat X', [X, X])
        repeat_y = self.register('repeat Y', [Y, Y])
        self.assertIs(report_robust(self.archive, fit_id, repeat_x, repeat_y,
                                    '1/3', '2/3', 1, 1)[1], True)
        self.assertIs(report_robust(self.archive, fit_id, repeat_x, repeat_y,
                                    '1/3', '7/10', 1, 1)[1], False)
        self.assertIs(report_robust(self.archive, fit_id, repeat_x, repeat_y,
                                    0, 1, 1, 1)[1], True)
        self.assertIs(report_robust(self.archive, fit_id, repeat_x, repeat_y,
                                    0, '101/100', 1, 1)[1], False)

    def test_oblique_basis_metric_and_translation(self):
        p = tuple(Fraction(value, 3) for value in (2, 1, -2))
        q = tuple(Fraction(value, 3) for value in (2, -2, 1))
        source = self.register('training', training_rows(OFFSET, p, q))
        fit_id = fit(self.archive, source, '1/2')
        left = self.register('P', [shifted(OFFSET, p)] * 2)
        right = self.register('Q', [shifted(OFFSET, q)] * 2)
        self.assertIs(report_robust(self.archive, fit_id, left, right,
                                    '1/3', '2/3', 1, 1)[1], True)

    def test_row_limits_and_budgets_reject_without_writing(self):
        source, fit_id = self._plane()
        one = self.register('one', [X])
        oversized = self.register('oversized', [X] * 13)
        before = self.counts()
        for arguments in (
            (fit_id, one, oversized, 0, 0, 0, 0),
            (fit_id, oversized, one, 0, 0, 0, 0),
            (fit_id, one, one, 0, 0, 1, 0),
            (fit_id, one, one, 0, 0, 0, 1),
            (fit_id, one, one, 0, 0, True, 0),
            (fit_id, one, one, 0, 0, 0, '1'),
            (fit_id, one, one, -1, 0, 0, 0),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(CoverageInputError):
                    report_robust(self.archive, *arguments)
                self.assertEqual(self.counts(), before)

    def test_budget_publication_rolls_back_and_retries(self):
        source, fit_id = self._plane()
        one = self.register('one', [X])
        before = self.counts()
        staged = []

        def interrupt():
            staged.append(self.counts())
            raise RuntimeError('report write interrupted')

        with self.assertRaisesRegex(PublicationError, 'report publication failed'):
            report_robust(self.archive, fit_id, one, one, 0, 0, 0, 0,
                          before_commit=interrupt)
        self.assertEqual(staged, [tuple(value + (1 if index in (3, 4) else 0)
                                        for index, value in enumerate(before))])
        self.reopen()
        self.assertEqual(self.counts(), before)
        report_id, passed = report_robust(self.archive, fit_id, one, one, 0, 0, 0, 0)
        view = read_result(self.archive, 'report', report_id)
        self.assertIs(view['passed'], passed)
        self.assertEqual((view['drop_left'], view['drop_right']), (0, 0))

    def test_corrections_keep_published_budgets_and_parent_ids(self):
        source, fit_id = self._plane()
        shared = self.register('shared', [X, ZERO])
        first = report_robust(self.archive, fit_id, shared, shared, 0, 0, 1, 0)
        second = report_robust(self.archive, fit_id, shared, shared, 0, 0, 0, 1)
        snapshot = [read_result(self.archive, 'report', identifier)
                    for identifier, _ in (first, second)]
        corrected = self.register('shared', [X, X], parent_rev=shared)
        before_reopen = self.counts()
        for _ in range(2):
            self.reopen()
            self.assertEqual(self.counts(), before_reopen)
            reread = [read_result(self.archive, 'report', identifier)
                      for identifier, _ in (first, second)]
            self.assertEqual(reread, snapshot)
            self.assertEqual([view['left_rev'] for view in reread], [shared, shared])
            self.assertEqual([view['drop_left'] for view in reread], [1, 0])
        self.assertEqual(self.archive.execute(
            'SELECT parent_rev FROM batch_revisions WHERE revision_id = ?',
            (corrected,)).fetchone()[0], shared)
