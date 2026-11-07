"""Ekspor parameter scorecard tanpa objek pickle atau data peminjam."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from src.scorecard import FACTOR, OFFSET


def export_bundle(bundle, path):
    model = bundle['model']
    woe = model.woe_
    payload = {
        'schema_version': 1,
        'target': 'bad=1; UCI source label 2',
        'seed': 2024,
        'selected': woe.selected_,
        'edges': {c: [float(v) if np.isfinite(v) else None for v in values]
                  for c, values in woe.edges_.items()},
        'medians': woe.medians_,
        'maps': woe.maps_,
        'coefficients': model.result_.params.drop('const').to_dict(),
        'intercept': float(model.result_.params['const']),
        'cutoff': bundle['cutoff'],
        'defaults': bundle['defaults'],
        'categories': bundle['categories'],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(payload, indent=2, allow_nan=False)+'\n').encode('utf-8'))
    return payload


class PortableScorecard:
    """Inferensi dari parameter tetap; tidak melakukan fit atau unduhan saat startup."""

    def __init__(self, payload):
        if payload['schema_version'] != 1:
            raise ValueError('Versi parameter scorecard tidak didukung.')
        self.payload = payload

    def predict_proba(self, frame):
        payload = self.payload
        logit = np.full(len(frame), payload['intercept'], dtype=float)
        for col in payload['selected']:
            if col in payload['edges']:
                edges = np.array(payload['edges'][col], dtype=float)
                edges[0], edges[-1] = -np.inf, np.inf
                values = pd.to_numeric(frame[col], errors='coerce').fillna(payload['medians'][col])
                labels = pd.cut(values, edges, include_lowest=True).astype(str)
            else:
                labels = frame[col].fillna('__MISSING__').astype(str)
            woe = labels.map(payload['maps'][col]).fillna(0).to_numpy()
            logit += payload['coefficients'][col]*woe
        probability = expit(logit)
        return np.column_stack([1-probability, probability])

    def points(self, frame):
        p = self.predict_proba(frame)[:, 1]
        return OFFSET - FACTOR*np.log(p/(1-p))
