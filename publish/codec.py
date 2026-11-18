from measure.rationals import format_rational, parse_rational


def encode_vector(vector):
    return [format_rational(value) for value in vector]


def _decode_vector(vector):
    return tuple(parse_rational(value) for value in vector)


def encode_fit_payload(unit, center, largest, basis):
    return {
        'unit': unit,
        'center': encode_vector(center),
        'largest': format_rational(largest),
        'basis': [encode_vector(column) for column in basis],
    }


def decode_fit_payload(payload):
    return {
        'unit': payload['unit'],
        'center': _decode_vector(payload['center']),
        'largest': parse_rational(payload['largest']),
        'basis': tuple(_decode_vector(column) for column in payload['basis']),
    }


def encode_report_payload(lo_left, lo_right, passed):
    return {
        'lo_left': format_rational(lo_left),
        'lo_right': format_rational(lo_right),
        'passed': passed,
    }


def encode_robust_report_payload(lo_left, lo_right, drop_left, drop_right, passed):
    return {
        'lo_left': format_rational(lo_left),
        'lo_right': format_rational(lo_right),
        'drop_left': drop_left,
        'drop_right': drop_right,
        'passed': passed,
    }


def decode_report_payload(payload):
    passed = payload['passed']
    if type(passed) is not bool:
        raise ValueError('stored coverage verdict must be boolean')
    return {
        'lo_left': parse_rational(payload['lo_left']),
        'lo_right': parse_rational(payload['lo_right']),
        'passed': passed,
    }


def decode_robust_report_payload(payload):
    if 'drop_left' not in payload or 'drop_right' not in payload:
        raise ValueError('stored robust report must carry both drop budgets')
    fields = decode_report_payload(payload)
    drops = []
    for name in ('drop_left', 'drop_right'):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('stored drop budget must be a non-boolean integer')
        drops.append(value)
    return {**fields, 'drop_left': drops[0], 'drop_right': drops[1]}
