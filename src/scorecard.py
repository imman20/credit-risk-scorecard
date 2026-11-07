"""Transformasi WoE yang hanya mempelajari data latih dan scorecard logistik."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from src.data.download import COLUMNS, NUMERIC

FACTOR = 20 / np.log(2)
OFFSET = 600 - FACTOR * np.log(50)


def probability_to_score(probability):
    p = np.clip(np.asarray(probability), 1e-12, 1 - 1e-12)
    return OFFSET + FACTOR * np.log((1 - p) / p)


def score_to_probability(score):
    return 1 / (1 + np.exp((np.asarray(score) - OFFSET) / FACTOR))


def validate_input(frame):
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Atribut belum lengkap: {sorted(missing)}')
    result = frame[COLUMNS].copy()
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col], errors='raise')
        if ((~np.isfinite(result[col])) | (result[col] <= 0)).any():
            raise ValueError(f'{col} harus angka positif hingga.')
    if (result.age < 18).any() or (result.age > 120).any():
        raise ValueError('Umur harus 18 sampai 120 tahun.')
    for col in set(COLUMNS) - set(NUMERIC):
        if result[col].isna().any() or (result[col].astype(str).str.len() == 0).any():
            raise ValueError(f'{col} wajib diisi.')
    return result


class WOETransformer(TransformerMixin, BaseEstimator):
    def __init__(self, bins=4, min_iv=0.02, smoothing=0.5):
        self.bins = bins
        self.min_iv = min_iv
        self.smoothing = smoothing

    def _labels(self, frame, col):
        if col in self.edges_:
            values = pd.to_numeric(frame[col], errors='coerce').fillna(self.medians_[col])
            return pd.cut(values, self.edges_[col], include_lowest=True).astype(str)
        return frame[col].fillna('__MISSING__').astype(str)

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        if set(np.unique(y)) != {0, 1} or len(y) != len(X):
            raise ValueError('WoE memerlukan kedua kelas dengan panjang yang sama.')
        self.columns_ = list(X.columns)
        self.edges_, self.medians_, self.maps_, self.iv_, tables = {}, {}, {}, {}, []
        self.fit_indices_ = tuple(X.index)
        for col in self.columns_:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.medians_[col] = float(X[col].median())
                edges = np.unique(X[col].quantile(np.linspace(0, 1, self.bins + 1)))
                self.edges_[col] = np.r_[-np.inf, edges[1:-1], np.inf]
            labels = self._labels(X, col)
            table = pd.DataFrame({'bin': labels.to_numpy(), 'bad': y}).groupby('bin').bad.agg(['sum', 'count'])
            k = len(table)
            bad = (table['sum'] + self.smoothing) / (y.sum() + self.smoothing * k)
            good = (table['count'] - table['sum'] + self.smoothing) / ((1-y).sum() + self.smoothing * k)
            table['woe'] = np.log(good / bad)
            table['iv_component'] = (good-bad) * table.woe
            table['attribute'] = col
            self.maps_[col] = table.woe.to_dict()
            self.iv_[col] = float(table.iv_component.sum())
            tables.append(table.reset_index())
        self.selected_ = [c for c in self.columns_ if self.iv_[c] >= self.min_iv]
        if not self.selected_:
            self.selected_ = [max(self.iv_, key=self.iv_.get)]
        self.table_ = pd.concat(tables, ignore_index=True)
        return self

    def transform(self, X):
        # Kategori yang tidak pernah dilihat diberi WoE netral, bukan dipelajari ulang.
        return pd.DataFrame({c: self._labels(X, c).map(self.maps_[c]).fillna(0.0)
                             for c in self.selected_}, index=X.index)


class Scorecard(ClassifierMixin, BaseEstimator):
    def __init__(self, min_iv=0.02):
        self.min_iv = min_iv

    def fit(self, X, y):
        self.woe_ = WOETransformer(min_iv=self.min_iv).fit(X, y)
        design = sm.add_constant(self.woe_.transform(X), has_constant='add')
        self.result_ = sm.GLM(np.asarray(y), design, family=sm.families.Binomial()).fit(maxiter=150)
        if not self.result_.converged or not np.isfinite(self.result_.params).all():
            raise RuntimeError('Logistik tidak konvergen; tidak menerbitkan scorecard.')
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        design = sm.add_constant(self.woe_.transform(X), has_constant='add')
        p = np.asarray(self.result_.predict(design))
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def points(self, X):
        woe = self.woe_.transform(X)
        points = woe.mul(-FACTOR * self.result_.params.drop('const'))
        points['base'] = OFFSET - FACTOR * self.result_.params['const']
        return points


def decision_cost(y, probability, cutoff):
    y = np.asarray(y)
    reject = np.asarray(probability) >= cutoff
    wrong_approve = int(((y == 1) & ~reject).sum())
    wrong_reject = int(((y == 0) & reject).sum())
    return {'cutoff': float(cutoff), 'cost': 5*wrong_approve + wrong_reject,
            'wrong_approve': wrong_approve, 'wrong_reject': wrong_reject,
            'tn': int(((y == 0) & ~reject).sum()), 'fp': wrong_reject,
            'fn': wrong_approve, 'tp': int(((y == 1) & reject).sum()),
            'approval_rate': float((~reject).mean())}


def select_cutoff(y, oof):
    candidates = np.r_[0, np.unique(oof), np.nextafter(max(oof), np.inf)]
    table = pd.DataFrame([decision_cost(y, oof, c) for c in candidates])
    # Bila biaya sama, pilih persetujuan terbanyak, lalu cutoff terkecil.
    return table.sort_values(['cost', 'approval_rate', 'cutoff'], ascending=[True, False, True]).iloc[0].to_dict()


def psi(reference, comparison, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins+1)))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, edges)[0] / len(reference)
    b = np.histogram(comparison, edges)[0] / len(comparison)
    a, b = np.maximum(a, 1e-6), np.maximum(b, 1e-6)
    return float(np.sum((b-a)*np.log(b/a)))
