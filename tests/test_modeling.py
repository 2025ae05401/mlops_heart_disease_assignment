import pandas as pd
from heartml.data import FEATURES
from heartml.modeling import build_models

def test_models_fit_and_predict():
    rows = [
        [55,130,220,150,1.0,1,2,0,0,1,0,2,3],
        [60,140,280,120,2.0,0,4,1,1,2,2,3,7],
        [45,120,190,170,0.0,1,1,0,0,0,0,1,3],
        [67,150,260,110,3.0,0,3,0,1,2,2,2,6],
    ]
    X = pd.DataFrame(rows, columns=FEATURES)
    y = [0,1,0,1]
    for model in build_models().values():
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(y)
        assert set(pred).issubset({0,1})
