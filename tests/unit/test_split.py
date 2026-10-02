import pandas as pd

from mlforge.data.split import split_churn_data


def sample_data(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "Churn": ["No"] * 140 + ["Yes"] * 60,
        }
    )


def test_split_preserves_all_rows_without_overlap() -> None:
    data = sample_data()
    splits = split_churn_data(data)

    assert len(splits.train) == 140
    assert len(splits.validation) == 30
    assert len(splits.test) == 30

    train_ids = set(splits.train["customerID"])
    validation_ids = set(splits.validation["customerID"])
    test_ids = set(splits.test["customerID"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(data["customerID"])


def test_split_is_reproducible() -> None:
    first = split_churn_data(sample_data())
    second = split_churn_data(sample_data())

    assert first.train["customerID"].tolist() == second.train["customerID"].tolist()
    assert (
        first.validation["customerID"].tolist()
        == second.validation["customerID"].tolist()
    )
    assert first.test["customerID"].tolist() == second.test["customerID"].tolist()


def test_split_is_stratified() -> None:
    data = sample_data()
    expected_rate = (data["Churn"] == "Yes").mean()
    splits = split_churn_data(data)

    for partition in (splits.train, splits.validation, splits.test):
        observed_rate = (partition["Churn"] == "Yes").mean()
        assert abs(observed_rate - expected_rate) <= 0.02
