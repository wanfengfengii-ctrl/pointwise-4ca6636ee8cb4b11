"""Exact decision procedure for the two-condition robust coverage report.

A condition holds n readings measured about a frozen training center. With a
drop budget d the operator may lose any d rows (losing fewer never lowers the
mean). The mean of the retained squared projections is nondecreasing in the
number retained, so a direction survives the budget exactly when the mean of
the n-d smallest squared projections reaches that position's threshold.

One dimensional frozen space
    There is a single projective direction, so the test is a direct
    comparison of rational numbers (the frozen basis column need not be a
    unit vector).

Two dimensional frozen space
    Directions form a real projective line. A condition with retained count
    k and threshold t accepts a direction u exactly when, for every subset S
    of k rows,

        sum_{i in S} (c_i . u)^2  >=  k * t * (u^T G u),

    where G is the Gram matrix of the (possibly non-orthogonal) frozen basis
    and c_i are the row coordinates with respect to that basis. Each
    inequality is a homogeneous quadratic form. On the projective line the
    sign of a quadratic form can change only at its real zeros, so the joint
    verdict is constant on every open arc between consecutive zeros of the
    finitely many forms. Testing every zero (closed thresholds) and one
    rational point of every open arc is therefore exact. Zeros are kept as
    exact numbers of the form a + b*sqrt(R); a rational midpoint is used for
    each open arc, and distinctness of zeros is decided algebraically.
"""

from fractions import Fraction
from functools import cmp_to_key
from itertools import combinations
from math import isqrt

from train.center import center_rows
from train.linear import dot


ROBUST_ENERGY = 'worst_retained_mean_squared_projection_about_training_center'


# ---------------------------------------------------------------------------
# exact arithmetic with numbers of the form a + b*sqrt(R)

def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def _sign_two(a, b, radicand: int) -> int:
    """Sign of a + b*sqrt(radicand) for rational a, b and integer radicand >= 0."""
    a = Fraction(a)
    b = Fraction(b)
    if radicand <= 0 or b == 0:
        return _sign(a)
    if a == 0:
        return _sign(b)
    if a > 0 and b > 0:
        return 1
    if a < 0 and b < 0:
        return -1
    squared = a * a - b * b * radicand
    if a > 0:
        return _sign(squared)
    return -_sign(squared)


def _sqrt_bounds(radicand: int, digits: int) -> tuple[Fraction, Fraction]:
    """Rational bounds lo <= sqrt(radicand) <= hi, exact by squaring."""
    scale = 10 ** digits
    root = isqrt(radicand * scale * scale)
    low = Fraction(root, scale)
    return low, low + Fraction(1, scale)


# ---------------------------------------------------------------------------
# projective slopes: a rational, a quadratic irrational a + b*sqrt(R), or INF

INF = ('infinity',)

_DIGITS = (16, 32, 64, 128, 256, 512, 1024, 2048)


def _finite(a, b, radicand) -> Fraction | tuple:
    a = Fraction(a)
    b = Fraction(b)
    radicand = int(radicand)
    if radicand <= 0 or b == 0:
        return a
    root = isqrt(radicand)
    if root * root == radicand:
        return a + b * root
    return ('root', a, b, radicand)


def _quadratic_equal(first, second) -> bool:
    """Exact equality of two quadratic irrationals a + b*sqrt(R)."""
    _, a1, b1, r1 = first
    _, a2, b2, r2 = second
    if r1 == r2:
        return a1 == a2 and b1 == b2
    product = r1 * r2
    root = isqrt(product)
    if root * root != product:
        # sqrt(r1) and sqrt(r2) are linearly independent over the rationals.
        return False
    # sqrt(r1) = root/r2 * sqrt(r2); reduce to one field.
    return a1 == a2 and b1 * Fraction(root, r2) == b2


def _interval(slope, digits: int) -> tuple[Fraction, Fraction]:
    if not isinstance(slope, tuple) or slope is INF:
        return slope, slope
    _, a, b, radicand = slope
    low, high = _sqrt_bounds(radicand, digits)
    if b > 0:
        return a + b * low, a + b * high
    return a + b * high, a + b * low


def _compare(first, second) -> int:
    """Exact comparison of two projective slopes, INF denoting the point at infinity."""
    if first is INF or second is INF:
        if first is INF and second is INF:
            return 0
        return 1 if first is INF else -1
    if isinstance(first, Fraction) and isinstance(second, Fraction):
        return _sign(first - second)
    if isinstance(first, tuple) and isinstance(second, tuple) and _quadratic_equal(first, second):
        return 0
    a1, b1, r1 = (first, Fraction(0), 0) if isinstance(first, Fraction) else first[1:4]
    a2, b2, r2 = (second, Fraction(0), 0) if isinstance(second, Fraction) else second[1:4]
    return _sign_three(a1 - a2, b1, r1, -b2, r2)


def _sign_radical_sum(b1, r1, b2, r2) -> int:
    """Sign of b1*sqrt(r1) + b2*sqrt(r2)."""
    b1, b2 = Fraction(b1), Fraction(b2)
    if r1 <= 0:
        return _sign_two(0, b2, r2)
    if r2 <= 0:
        return _sign_two(0, b1, r1)
    if b1 == 0:
        return _sign(b2)
    if b2 == 0:
        return _sign(b1)
    if _sign(b1) == _sign(b2):
        return _sign(b1)
    gap = _sign(b1 * b1 * r1 - b2 * b2 * r2)
    if gap == 0:
        return 0
    return _sign(b1) if gap > 0 else _sign(b2)


def _sign_three(a, b1, r1, b2, r2) -> int:
    """Sign of a + b1*sqrt(r1) + b2*sqrt(r2) for rational a, b and integer r >= 0."""
    a = Fraction(a)
    radical_sign = _sign_radical_sum(b1, r1, b2, r2)
    if radical_sign == 0:
        return _sign(a)
    if a == 0:
        return radical_sign
    if _sign(a) == radical_sign:
        return _sign(a)
    # Opposite signs: compare a^2 with v^2, where v = b1 sqrt r1 + b2 sqrt r2.
    b1, b2 = Fraction(b1), Fraction(b2)
    r1, r2 = int(r1), int(r2)
    squared = (b1 * b1 * r1 + b2 * b2 * r2, 2 * b1 * b2, r1 * r2)
    difference = _sign_two(a * a - squared[0], -squared[1], squared[2])
    if difference == 0:
        return 0
    return _sign(a) if difference > 0 else radical_sign



def _slope_key(slope):
    return cmp_to_key(_compare)(slope)


def _distinct(sorted_slopes):
    result = []
    for slope in sorted_slopes:
        if not result or _compare(result[-1], slope) != 0:
            result.append(slope)
    return result


def _between(low, high) -> Fraction:
    """A rational strictly between two distinct finite slopes."""
    for digits in _DIGITS:
        _, low_high = _interval(low, digits)
        high_low, _ = _interval(high, digits)
        if low_high < high_low:
            return (low_high + high_low) / 2
    raise ArithmeticError('adjacent algebraic directions were not separated')


def _outside(slope, above: bool) -> Fraction:
    """A rational strictly to one side of a finite slope."""
    low, high = _interval(slope, 64)
    return high + 1 if above else low - 1


# ---------------------------------------------------------------------------
# quadratic forms and their real projective zeros

def _roots(form) -> list:
    """Real projective zeros of a + 2 b t + c t^2 in the slope chart."""
    a, b, c = form
    if a == 0 and b == 0 and c == 0:
        return []
    if c == 0:
        # The coefficient at the point at infinity is c, so infinity is a
        # projective zero here; the remaining zero is finite.
        roots = [INF]
        if b != 0:
            roots.append(_finite(-a / (2 * b), 0, 1))
        return roots
    discriminant = b * b - a * c
    if discriminant < 0:
        return []
    radicand = discriminant.numerator * discriminant.denominator
    base = -b / c
    coefficient = Fraction(1) / (discriminant.denominator * c)
    return [_finite(base, coefficient, radicand), _finite(base, -coefficient, radicand)]


def _forms(coordinates, metric, retained: int, threshold: Fraction):
    """Quadratic forms whose simultaneous nonnegativity characterises a condition."""
    g11, g12, g22 = metric[0][0], metric[0][1], metric[1][1]
    seen = set()
    forms = []
    for subset in combinations(range(len(coordinates)), retained):
        a = b = c = Fraction(0)
        for index in subset:
            x, y = coordinates[index]
            a += x * x
            b += x * y
            c += y * y
        form = (a - retained * threshold * g11,
                b - retained * threshold * g12,
                c - retained * threshold * g22)
        if form not in seen:
            seen.add(form)
            forms.append(form)
    return forms


def _metric(basis):
    return tuple(tuple(dot(first, second) for second in basis) for first in basis)


# ---------------------------------------------------------------------------
# retained mean evaluated exactly at a candidate direction

def _retained_ok(slope, coordinates, metric, retained: int, threshold: Fraction) -> bool:
    g11, g12, g22 = metric[0][0], metric[0][1], metric[1][1]
    if slope is INF:
        first = (Fraction(0), Fraction(0), 1)
        second = (Fraction(1), Fraction(0), 1)
    else:
        first = (Fraction(1), Fraction(0), 1)
        if isinstance(slope, Fraction):
            second = (slope, Fraction(0), 1)
        else:
            _, base, coefficient, radicand = slope
            second = (base, coefficient, radicand)
    (u1a, u1b, _), (u2a, u2b, radicand) = first, second

    norm_a = (g11 * (u1a * u1a + u1b * u1b * radicand)
              + 2 * g12 * (u1a * u2a + u1b * u2b * radicand)
              + g22 * (u2a * u2a + u2b * u2b * radicand))
    norm_b = (2 * g11 * u1a * u1b
              + 2 * g12 * (u1a * u2b + u1b * u2a)
              + 2 * g22 * u2a * u2b)

    squared = []
    for x, y in coordinates:
        const = x * u1a + y * u2a
        radical = x * u1b + y * u2b
        squared.append((const * const + radical * radical * radicand, 2 * const * radical))

    def compare(left, right):
        return _sign_two(left[0] - right[0], left[1] - right[1], radicand)

    squared.sort(key=cmp_to_key(compare))
    total_a = total_b = Fraction(0)
    for a, b in squared[:retained]:
        total_a += a
        total_b += b
    return _sign_two(total_a - retained * threshold * norm_a,
                     total_b - retained * threshold * norm_b, radicand) >= 0


# ---------------------------------------------------------------------------
# decision

def _candidates(forms) -> list:
    finite = []
    for group in forms:
        for form in group:
            for root in _roots(form):
                if root is not INF:
                    finite.append(root)
    finite.sort(key=_slope_key)
    finite = _distinct(finite)

    if not finite:
        return [Fraction(0), INF]
    candidates = list(finite) + [INF]
    candidates.extend(_between(low, high) for low, high in zip(finite, finite[1:]))
    candidates.append(_outside(finite[0], above=False))
    candidates.append(_outside(finite[-1], above=True))
    return candidates


def _two_dimensional(basis, left, right, thresholds, drops) -> bool:
    metric = _metric(basis)
    coordinates = [tuple((dot(row, basis[0]), dot(row, basis[1])) for row in rows)
                   for rows in (left, right)]
    retained = [len(coordinates[0]) - drops[0], len(coordinates[1]) - drops[1]]
    forms = [_forms(coordinates[0], metric, retained[0], thresholds[0]),
             _forms(coordinates[1], metric, retained[1], thresholds[1])]

    return any(_retained_ok(slope, coordinates[0], metric, retained[0], thresholds[0])
               and _retained_ok(slope, coordinates[1], metric, retained[1], thresholds[1])
               for slope in _candidates(forms))


def _one_dimensional(basis, left, right, thresholds, drops) -> bool:
    direction = basis[0]
    norm = dot(direction, direction)
    for rows, threshold, drop in zip((left, right), thresholds, drops):
        retained = len(rows) - drop
        values = sorted(dot(row, direction) ** 2 for row in rows)
        if sum(values[:retained], Fraction(0)) < retained * threshold * norm:
            return False
    return True


def robust_verdict(basis, center, left_rows, right_rows, lo_left, lo_right,
                   drop_left, drop_right) -> bool:
    """Decide whether one frozen-space unit axis serves both conditions."""
    left = center_rows(left_rows, center)
    right = center_rows(right_rows, center)
    thresholds = (lo_left, lo_right)
    drops = (drop_left, drop_right)
    if len(basis) == 1:
        return _one_dimensional(basis, left, right, thresholds, drops)
    return _two_dimensional(basis, left, right, thresholds, drops)
