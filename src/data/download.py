"""Unduh arsip asli UCI dan validasi 20 atribut serta label."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import requests

URL = 'https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip'
COLUMNS = ['checking', 'duration', 'history', 'purpose', 'amount', 'savings',
           'employment', 'installment', 'personal_status_sex', 'debtors', 'residence',
           'property', 'age', 'other_plans', 'housing', 'existing_credits', 'job',
           'dependents', 'telephone', 'foreign_worker']
NUMERIC = ['duration', 'amount', 'installment', 'residence', 'age',
           'existing_credits', 'dependents']


def map_target(labels):
    if not labels.isin([1, 2]).all():
        raise ValueError('Target UCI harus 1=good atau 2=bad.')
    return (labels == 2).astype(int).rename('default')


def load(path='data/raw/german.data'):
    data = pd.read_csv(path, sep=r'\s+', names=COLUMNS + ['label'])
    if data.shape != (1000, 21) or data.isna().any().any():
        raise ValueError('Skema UCI tidak sesuai: harus 1000 baris lengkap.')
    map_target(data.label)
    return data


def download():
    folder = Path('data/raw')
    folder.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=90)
    response.raise_for_status()
    payload = response.content
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in ['german.data', 'german.doc']:
            (folder / name).write_bytes(archive.read(name))
    data = load()
    Path('data/sample').mkdir(parents=True, exist_ok=True)
    data.head(12).to_csv('data/sample/german_sample.csv', index=False)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/source.json').write_text(json.dumps({
        'url': URL, 'sha256_zip': hashlib.sha256(payload).hexdigest(),
        'rows': len(data), 'target': {'1': 'good', '2': 'bad/default=1'}
    }, indent=2), encoding='utf-8')
    print(f'UCI: {len(data)} baris tervalidasi.')


if __name__ == '__main__':
    download()
