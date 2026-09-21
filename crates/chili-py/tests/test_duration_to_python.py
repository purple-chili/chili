"""Duration -> datetime.timedelta keeps its sign."""

from datetime import timedelta

from chili import ChiliEngine


def test_negative_durations_keep_their_sign():
    e = ChiliEngine(pepper=True)
    assert e.eval("0D00:00:01") == timedelta(seconds=1)
    assert e.eval("neg 0D00:00:01") == timedelta(seconds=-1)
    assert e.eval("neg 0D00:00:01.500000") == timedelta(seconds=-1.5)
    assert e.eval("neg 1D00:00:01") == timedelta(days=-1, seconds=-1)
    assert e.eval("1D02:03:04.000005") == timedelta(
        days=1, hours=2, minutes=3, seconds=4, microseconds=5
    )
    e.shutdown()
