"""Tests for the restricted pickle loader, including a booby-trapped file."""

import os
import pickle
from collections import OrderedDict

import numpy as np
import pytest

import safe_pickle


class _Trap:
    """When a normal pickle load rebuilds this, it runs os.mkdir(path)."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (os.mkdir, (str(self.path),))


class _PlainCustomClass:
    def __init__(self):
        self.value = 1


def test_the_trap_really_runs_with_a_normal_load_so_the_test_below_means_something(tmp_path):
    marker = tmp_path / "ran_with_plain_pickle"
    data = pickle.dumps(_Trap(marker))
    pickle.loads(data)  # the unsafe way
    assert marker.exists()


def test_booby_trapped_pickle_is_refused_and_does_not_run(tmp_path):
    marker = tmp_path / "must_not_exist"
    data = pickle.dumps(_Trap(marker))
    with pytest.raises(safe_pickle.UnsafePickleError):
        safe_pickle.loads(data)
    assert not marker.exists()


def test_a_trap_inside_a_normal_looking_list_is_refused(tmp_path):
    marker = tmp_path / "must_not_exist"
    data = pickle.dumps([OrderedDict(a=1), {"x": _Trap(marker)}])
    with pytest.raises(safe_pickle.UnsafePickleError):
        safe_pickle.loads(data)
    assert not marker.exists()


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_plain_data_in_the_real_layout_loads_unchanged(protocol):
    original = [
        OrderedDict(
            group3_holder_num=150,
            group3_time_span=412.5,
            mint_ts=1733702400,
            mint_address="SAMPLE0001",
            ts=np.arange(1800, dtype=np.float32).reshape(360, 5),
            ts_len=29,
        )
    ]
    loaded = safe_pickle.loads(pickle.dumps(original, protocol=protocol))
    assert isinstance(loaded[0], OrderedDict)
    assert loaded[0]["group3_holder_num"] == 150
    assert loaded[0]["mint_address"] == "SAMPLE0001"
    assert loaded[0]["ts"].dtype == np.float32
    assert loaded[0]["ts"].shape == (360, 5)
    np.testing.assert_array_equal(loaded[0]["ts"], original[0]["ts"])


def test_sets_tuples_bytes_and_numpy_numbers_load():
    original = {"s": {1, 2}, "f": frozenset([3]), "t": (1, "a"), "b": b"xyz", "n": np.float64(1.5), "i": np.int32(7)}
    loaded = safe_pickle.loads(pickle.dumps(original))
    assert loaded["s"] == {1, 2} and loaded["f"] == frozenset([3]) and loaded["t"] == (1, "a")
    assert loaded["b"] == b"xyz" and loaded["n"] == 1.5 and loaded["i"] == 7


@pytest.mark.parametrize(
    "thing",
    [
        os.system,
        eval,
        getattr,
        open,
    ],
)
def test_dangerous_functions_are_refused_by_name(thing):
    class Bad:
        def __reduce__(self):
            return (thing, ("x",))

    with pytest.raises(safe_pickle.UnsafePickleError):
        safe_pickle.loads(pickle.dumps(Bad()))


def test_a_custom_class_is_refused():
    with pytest.raises(safe_pickle.UnsafePickleError):
        safe_pickle.loads(pickle.dumps(_PlainCustomClass()))


def test_a_pandas_table_is_refused():
    pandas = pytest.importorskip("pandas")
    with pytest.raises(safe_pickle.UnsafePickleError):
        safe_pickle.loads(pickle.dumps(pandas.DataFrame({"a": [1]})))


def test_load_file_reads_a_file_from_disk(tmp_path):
    path = tmp_path / "x.pkl"
    path.write_bytes(pickle.dumps({"a": np.arange(3)}))
    np.testing.assert_array_equal(safe_pickle.load_file(path)["a"], np.arange(3))


def test_the_error_message_names_what_was_refused_and_nothing_else(tmp_path):
    data = pickle.dumps(_Trap(tmp_path / "m"))
    with pytest.raises(safe_pickle.UnsafePickleError) as error:
        safe_pickle.loads(data)
    assert "refused" in str(error.value)
    assert str(tmp_path) not in str(error.value)
