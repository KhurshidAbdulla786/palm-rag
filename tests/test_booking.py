from datetime import date

from app.services.booking import merge_and_validate

TODAY = date(2026, 11, 16)


def test_merges_new_values_into_existing_state() -> None:
    state, errors = merge_and_validate(
        {"name": "Asha Rai"}, {"email": "asha@example.com", "date": None}, TODAY
    )
    assert state == {"name": "Asha Rai", "email": "asha@example.com"}
    assert errors == []


def test_rejects_invalid_email_and_past_date() -> None:
    state, errors = merge_and_validate(
        {}, {"email": "nope", "date": "2020-01-01"}, TODAY
    )
    assert state == {}
    assert len(errors) == 2


def test_normalizes_time() -> None:
    state, _ = merge_and_validate({}, {"time": "9:30"}, TODAY)
    assert state["time"] == "09:30"
