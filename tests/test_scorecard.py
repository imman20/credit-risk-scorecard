import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from streamlit.testing.v1 import AppTest

from src.data.download import COLUMNS, NUMERIC, map_target
from src.pipeline import run
from src.scorecard import (Scorecard, WOETransformer, decision_cost, probability_to_score,
                           score_to_probability, select_cutoff, validate_input)


@pytest.fixture
def fixture_credit():
    # Data buatan hanya untuk unit/smoke test, tidak untuk angka portofolio.
    rng = np.random.default_rng(10)
    n = 240
    frame = pd.DataFrame({c: rng.choice(['A', 'B', 'C'], n) for c in COLUMNS})
    for col in NUMERIC:
        frame[col] = rng.integers(1, 25, n)
    frame['age'] = rng.integers(18, 75, n)
    frame['personal_status_sex'] = rng.choice(['A91', 'A92', 'A93'], n)
    p = 1/(1+np.exp(-(.8*(frame.checking == 'A') + .07*frame.duration - 1.5)))
    frame['label'] = np.where(rng.random(n) < p, 2, 1)
    return frame


def test_target_mapping_and_rejection():
    assert map_target(pd.Series([1, 2, 1])).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        map_target(pd.Series([0, 1]))


def test_fit_ignores_holdout_distribution_and_labels(fixture_credit):
    train, holdout = fixture_credit.iloc[:180], fixture_credit.iloc[180:].copy()
    transformer = WOETransformer().fit(train[COLUMNS], map_target(train.label))
    before_edges = {k: v.copy() for k, v in transformer.edges_.items()}
    before_maps = {k: v.copy() for k, v in transformer.maps_.items()}
    holdout['amount'] = 1e9
    holdout['checking'] = 'UNSEEN'
    transformed = transformer.transform(holdout[COLUMNS])
    assert transformer.fit_indices_ == tuple(train.index)
    assert set(transformer.fit_indices_).isdisjoint(holdout.index)
    assert transformer.maps_ == before_maps
    for col in before_edges:
        np.testing.assert_array_equal(before_edges[col], transformer.edges_[col])
    assert np.isfinite(transformed).all().all()


def test_cv_refits_transformations_per_fold(fixture_credit, monkeypatch):
    calls = []
    original = WOETransformer.fit

    def spy(self, X, y):
        calls.append(set(X.index))
        return original(self, X, y)

    monkeypatch.setattr(WOETransformer, 'fit', spy)
    X, y = fixture_credit[COLUMNS], map_target(fixture_credit.label)
    cv = StratifiedKFold(3, shuffle=True, random_state=2024)
    pred = cross_val_predict(Scorecard(), X, y, cv=cv, method='predict_proba')
    assert len(calls) == 3
    for fitted_ids, (train, valid) in zip(calls, cv.split(X, y)):
        assert fitted_ids == set(X.iloc[train].index)
        assert fitted_ids.isdisjoint(X.iloc[valid].index)
    assert np.isfinite(pred).all()


def test_unknown_category_is_neutral():
    X = pd.DataFrame({'category': ['a', 'a', 'b', 'b', None, None]})
    t = WOETransformer(min_iv=0).fit(X, [0, 0, 1, 1, 0, 1])
    assert t.transform(pd.DataFrame({'category': ['new']})).iloc[0, 0] == 0
    assert np.isfinite(t.transform(pd.DataFrame({'category': [None]}))).all().all()


def test_points_odds_orientation_and_probability(fixture_credit):
    model = Scorecard().fit(fixture_credit[COLUMNS], map_target(fixture_credit.label))
    p = model.predict_proba(fixture_credit[COLUMNS])[:, 1]
    score = probability_to_score(p)
    np.testing.assert_allclose(model.points(fixture_credit[COLUMNS]).sum(axis=1), score)
    np.testing.assert_allclose(score_to_probability(score), p)
    assert probability_to_score(1/51) == pytest.approx(600)
    assert probability_to_score(1/101) == pytest.approx(620)
    assert probability_to_score(.1) > probability_to_score(.2)


def test_cost_and_cutoff_boundary():
    result = decision_cost([1, 0, 1, 0], [.1, .8, .9, .1], .8)
    assert result['cost'] == 6
    assert result['wrong_approve'] == result['wrong_reject'] == 1
    assert result['approval_rate'] == .5
    y, oof = [1, 0, 1, 0], np.array([.2, .3, .8, .1])
    selected = select_cutoff(y, oof)
    assert selected['cost'] == min(decision_cost(y, oof, c)['cost'] for c in [0, .1, .2, .3, .8, 1])


def test_input_validation(fixture_credit):
    row = fixture_credit[COLUMNS].iloc[:1].copy()
    validate_input(row)
    row['age'] = 0
    with pytest.raises(ValueError):
        validate_input(row)
    with pytest.raises(ValueError):
        validate_input(row.drop(columns='checking'))


def test_pipeline_smoke(fixture_credit, tmp_path):
    result = run(fixture_credit, tmp_path)
    assert result['train_n'] + result['test_n'] == len(fixture_credit)
    assert result['logistic_converged']
    assert result['score_probability_max_error'] < 1e-8
    assert (tmp_path/'reports/hasil.json').exists()
    assert (tmp_path/'assets/evaluation.png').stat().st_size > 1000


def test_streamlit_smoke_and_prediction():
    app = AppTest.from_file('app/streamlit_app.py').run(timeout=30)
    assert not app.exception
    if app.button:
        app.button[0].click().run(timeout=30)
        assert not app.exception
        assert len(app.metric) == 3
        app.text_input[0].set_value('CATEGORY_NOT_SEEN').run()
        app.button[0].click().run(timeout=30)
        assert not app.exception
        assert any('tidak dikenal' in warning.value for warning in app.warning)
    else:
        assert any('run_all.py' in message.value for message in app.info)
