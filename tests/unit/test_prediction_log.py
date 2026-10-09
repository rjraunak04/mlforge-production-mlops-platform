from mlforge.monitoring.prediction_log import (
    append_prediction_event,
    build_prediction_event,
    read_prediction_events,
)


def test_prediction_events_round_trip(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    event = build_prediction_event(
        request_id="req-1",
        model_name="model",
        model_version="1",
        prediction="Yes",
        churn_probability=0.8,
        features={"tenure": 3},
    )
    append_prediction_event(path, event)
    loaded = read_prediction_events(path)
    assert loaded == [event]
    assert loaded[0].features == {"tenure": 3}


def test_missing_prediction_log_is_empty(tmp_path) -> None:
    assert read_prediction_events(tmp_path / "missing.jsonl") == []
