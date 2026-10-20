import pandas as pd
import pytest
from heartml.data import FEATURES, TARGET, validate_dataset
from heartml.modeling import make_preprocessor

def test_feature_schema():
    assert len(FEATURES) == 13
    assert TARGET == "target"

def test_validate_dataset():
    row = {f: 1 for f in FEATURES}
    row[TARGET] = 1
    row["extra"] = "ignored"
    result = validate_dataset(pd.DataFrame([row]))
    assert result.shape == (1, 14)
    assert "extra" not in result.columns

def test_nonbinary_target_rejected():
    row = {f: 1 for f in FEATURES}
    row[TARGET] = 2
    with pytest.raises(ValueError):
        validate_dataset(pd.DataFrame([row]))

def test_preprocessor_imputes_missing():
    row = {"age":50,"trestbps":None,"chol":200,"thalach":150,"oldpeak":1.2,
           "sex":1,"cp":2,"fbs":0,"restecg":1,"exang":0,"slope":2,"ca":0,"thal":3}
    transformed = make_preprocessor().fit_transform(pd.DataFrame([row]))
    assert transformed.shape[0] == 1
