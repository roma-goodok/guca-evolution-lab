# tests/ga/test_activity_scheme_helper.py
from guca.ga.toolbox import _activity_scheme

def test_activity_scheme_compact():
    assert _activity_scheme([False, False, True, True, False]) == "2xx1"
