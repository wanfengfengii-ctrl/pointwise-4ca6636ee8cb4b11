import json
import sqlite3
from dataclasses import dataclass
from fractions import Fraction

from publish.codec import (decode_report_payload, decode_robust_report_payload,
                           encode_report_payload, encode_robust_report_payload)
from publish.links import load_report_links, result_key


class UnknownReport(LookupError):
    pass


@dataclass(frozen=True)
class ReportRecord:
    report_id: str
    fit_id: str
    left_rev: int
    right_rev: int
    lo_left: Fraction
    lo_right: Fraction
    passed: bool
    drop_left: int | None = None
    drop_right: int | None = None

    @property
    def is_robust(self) -> bool:
        return self.drop_left is not None


def insert_report(archive: sqlite3.Connection, lo_left, lo_right, passed: bool) -> int:
    payload = encode_report_payload(lo_left, lo_right, passed)
    cursor = archive.execute(
        'INSERT INTO reports (payload) VALUES (?)',
        (json.dumps(payload, separators=(',', ':')),),
    )
    return cursor.lastrowid


def insert_robust_report(archive: sqlite3.Connection, lo_left, lo_right,
                         drop_left, drop_right, passed: bool) -> int:
    payload = encode_robust_report_payload(lo_left, lo_right, drop_left, drop_right, passed)
    cursor = archive.execute(
        'INSERT INTO reports (payload) VALUES (?)',
        (json.dumps(payload, separators=(',', ':')),),
    )
    return cursor.lastrowid


def load_report(archive: sqlite3.Connection, report_id: str) -> ReportRecord:
    key = result_key(report_id, 'report')
    row = archive.execute('SELECT payload FROM reports WHERE report_id = ?', (key,)).fetchone()
    if row is None:
        raise UnknownReport(f'unknown report {report_id!r}')
    fit_key, left_rev, right_rev = load_report_links(archive, key)
    payload = json.loads(row[0])
    if 'drop_left' in payload or 'drop_right' in payload:
        fields = decode_robust_report_payload(payload)
    else:
        fields = decode_report_payload(payload)
    return ReportRecord(report_id=report_id, fit_id=f'fit:{fit_key}',
                        left_rev=left_rev, right_rev=right_rev, **fields)
