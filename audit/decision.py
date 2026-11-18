from fractions import Fraction

from audit.criterion import ThresholdPair, build_threshold_pair
from audit.energy import compute_energy_form
from audit.request import resolve_request
from publish.commit import publish_report


def _nonnegative_radical(p: Fraction, q: Fraction, radicand: Fraction) -> bool:
    """Compare p + q sqrt(radicand) with zero by squaring only opposite signs."""
    if radicand == 0 or q == 0:
        return p >= 0
    if q > 0:
        return p >= 0 or q * q * radicand >= p * p
    return p >= 0 and p * p >= q * q * radicand


def _covers_at_finite_boundary(boundary, other) -> bool:
    a, b = boundary[0]
    c = boundary[1][1]
    u, v = other[0]
    w = other[1][1]
    if c == 0:
        # The finite root has homogeneous coordinates (2b, -a) when b is nonzero.
        if b == 0:
            return False
        return u * (2 * b)**2 - 4 * v * b * a + w * a**2 >= 0

    discriminant = b * b - a * c
    if discriminant < 0:
        return False
    # Substitute (c, -b ± sqrt(discriminant)) into the other form.
    p = u * c * c - 2 * v * c * b + w * (b * b + discriminant)
    q = 2 * (v * c - w * b)
    return (_nonnegative_radical(p, q, discriminant)
            or _nonnegative_radical(p, -q, discriminant))


def has_common_direction(pair: ThresholdPair) -> bool:
    left, right = pair.left, pair.right
    if left[0][0] >= 0 and right[0][0] >= 0:
        return True
    if len(left) == 1:
        return False
    if left[1][1] >= 0 and right[1][1] >= 0:
        return True

    # On the projective line a path from any feasible point to the rejected
    # first axis must encounter a feasible zero of a nonzero form.
    return (_covers_at_finite_boundary(left, right)
            or _covers_at_finite_boundary(right, left))


def report(archive, fit_id, left_rev, right_rev, lo_left, lo_right, *, before_commit=None):
    request = resolve_request(archive, fit_id, left_rev, right_rev, lo_left, lo_right)
    center, basis = request.fit.center, request.fit.basis
    left_form = compute_energy_form(request.left.rows, center, basis)
    right_form = compute_energy_form(request.right.rows, center, basis)
    pair = build_threshold_pair(basis, left_form, right_form, request.lo_left, request.lo_right)
    passed = has_common_direction(pair)
    report_id = publish_report(
        archive, request.fit.fit_id, request.left.revision_id, request.right.revision_id,
        request.lo_left, request.lo_right, passed, before_commit=before_commit,
    )
    return report_id, passed
