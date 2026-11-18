from dataclasses import dataclass
from fractions import Fraction

from measure.catalog import BatchRevision, load_revision
from measure.rationals import parse_rational
from publish.fit_records import FitRecord, load_fit


class CoverageInputError(ValueError):
    pass


@dataclass(frozen=True)
class CoverageRequest:
    fit: FitRecord
    left: BatchRevision
    right: BatchRevision
    lo_left: Fraction
    lo_right: Fraction


MAX_CONDITION_ROWS = 12


@dataclass(frozen=True)
class RobustRequest:
    fit: FitRecord
    left: BatchRevision
    right: BatchRevision
    lo_left: Fraction
    lo_right: Fraction
    drop_left: int
    drop_right: int


def _threshold(value, position):
    try:
        threshold = parse_rational(value)
    except ValueError as exc:
        raise CoverageInputError(f'{position} threshold: {exc}') from exc
    if threshold < 0:
        raise CoverageInputError(f'{position} threshold must be nonnegative')
    return threshold


def resolve_request(archive, fit_id, left_rev, right_rev, lo_left, lo_right) -> CoverageRequest:
    left_minimum = _threshold(lo_left, 'left')
    right_minimum = _threshold(lo_right, 'right')
    frozen, left, right = _resolve_bound(archive, fit_id, left_rev, right_rev)
    return CoverageRequest(frozen, left, right, left_minimum, right_minimum)


def _budget(value, position, row_count) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CoverageInputError(f'{position} drop budget must be a non-boolean integer')
    if not 0 <= value < row_count:
        raise CoverageInputError(
            f'{position} drop budget must satisfy 0 <= budget < {row_count} rows')
    return value


def resolve_robust_request(archive, fit_id, left_rev, right_rev, lo_left, lo_right,
                           drop_left, drop_right) -> RobustRequest:
    left_minimum = _threshold(lo_left, 'left')
    right_minimum = _threshold(lo_right, 'right')
    if isinstance(drop_left, bool) or not isinstance(drop_left, int):
        raise CoverageInputError('left drop budget must be a non-boolean integer')
    if isinstance(drop_right, bool) or not isinstance(drop_right, int):
        raise CoverageInputError('right drop budget must be a non-boolean integer')
    frozen, left, right = _resolve_bound(archive, fit_id, left_rev, right_rev)
    for position, batch in (('left', left), ('right', right)):
        if not 1 <= len(batch.rows) <= MAX_CONDITION_ROWS:
            raise CoverageInputError(
                f'{position} condition must hold between 1 and {MAX_CONDITION_ROWS} rows')
    left_budget = _budget(drop_left, 'left', len(left.rows))
    right_budget = _budget(drop_right, 'right', len(right.rows))
    return RobustRequest(frozen, left, right, left_minimum, right_minimum,
                         left_budget, right_budget)


def _resolve_bound(archive, fit_id, left_rev, right_rev):
    try:
        frozen = load_fit(archive, fit_id)
    except (LookupError, ValueError) as exc:
        raise CoverageInputError(f'fit source: {exc}') from exc
    try:
        left = load_revision(archive, left_rev)
        right = load_revision(archive, right_rev)
    except LookupError as exc:
        raise CoverageInputError(f'condition source: {exc}') from exc

    if left.unit != frozen.unit:
        raise CoverageInputError('left condition unit must equal the training unit')
    if right.unit != frozen.unit:
        raise CoverageInputError('right condition unit must equal the training unit')
    return frozen, left, right
