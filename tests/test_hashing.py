"""Tests for canonicalization and call-site hashing.

The single most important property of the whole library is implemented
in `agentreplay.hashing`: the *same* logical call must produce the same
call-site ID, regardless of cosmetic differences (dict key ordering,
non-deterministic UUIDs, embedded timestamps).
"""
from __future__ import annotations

from agentreplay.hashing import canonical_json, canonicalize, diff_keys, hash_call_site


def test_canonicalize_sorts_dict_keys():
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert canonicalize(a) == canonicalize(b)


def test_canonicalize_strips_nondeterministic_keys():
    a = {"model": "gpt", "request_id": "abc", "messages": []}
    b = {"model": "gpt", "messages": [], "request_id": "xyz"}
    assert canonicalize(a) == canonicalize(b)


def test_canonicalize_redacts_uuids_in_strings():
    a = {"prompt": "user id is 11111111-2222-3333-4444-555555555555 today"}
    b = {"prompt": "user id is 99999999-8888-7777-6666-555555555555 today"}
    assert canonicalize(a) == canonicalize(b)


def test_canonicalize_redacts_iso8601_timestamps():
    a = {"ts": "2026-07-12T13:00:00Z"}
    b = {"ts": "2026-07-12T14:30:00Z"}
    assert canonicalize(a) == canonicalize(b)


def test_canonical_json_is_stable():
    import json

    a = {"b": 1, "a": 2, "nested": {"y": [1, 2], "x": "s"}}
    encoded = canonical_json(a)
    decoded = json.loads(encoded)
    # Sorted-keys property
    assert encoded.index('"a"') < encoded.index('"b"')
    assert decoded == {"a": 2, "b": 1, "nested": {"x": "s", "y": [1, 2]}}


def test_hash_call_site_is_deterministic():
    """The SAME logical request must produce the SAME call-site ID."""
    request = {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4"}

    # Cosmetic differences must not change the hash.
    a = hash_call_site("step:0", request, call_type="llm")
    b = hash_call_site("step:0", {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}, call_type="llm")
    assert a == b


def test_hash_call_site_changes_with_step_id():
    request = {"messages": [{"role": "user", "content": "hello"}]}
    a = hash_call_site("step:0", request, call_type="llm")
    b = hash_call_site("step:1", request, call_type="llm")
    assert a != b


def test_hash_call_site_changes_with_call_type():
    request = {"messages": [{"role": "user", "content": "hello"}]}
    a = hash_call_site("step:0", request, call_type="llm")
    b = hash_call_site("step:0", request, call_type="tool")
    assert a != b


def test_hash_call_site_strips_request_id():
    """Non-deterministic keys must NOT affect the call-site ID."""
    request_a = {"model": "gpt", "messages": [], "request_id": "abc"}
    request_b = {"model": "gpt", "messages": [], "request_id": "xyz"}
    a = hash_call_site("step:0", request_a, call_type="llm")
    b = hash_call_site("step:0", request_b, call_type="llm")
    assert a == b


def test_diff_keys_empty_for_identical():
    a = {"x": 1, "y": [1, 2]}
    b = {"y": [1, 2], "x": 1}
    assert list(diff_keys(a, b)) == []


def test_diff_keys_reports_value_change():
    a = {"x": 1, "y": [1, 2]}
    b = {"x": 2, "y": [1, 2]}
    diffs = list(diff_keys(a, b))
    assert "x" in diffs


def test_diff_keys_reports_missing_keys():
    a = {"x": 1}
    b = {"x": 1, "y": 2}
    diffs = list(diff_keys(a, b))
    assert "y" in diffs


def test_diff_keys_reports_list_length_change():
    a = {"items": [1, 2]}
    b = {"items": [1, 2, 3]}
    diffs = list(diff_keys(a, b))
    assert "items[2]" in diffs
