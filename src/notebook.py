"""Bangun dan eksekusi notebook audit dari artefak pipeline yang sama."""
import hashlib
import json
import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def main():
    root = Path.cwd()
    manifest = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in ['reports', 'assets'] for p in sorted((root/folder).glob('*')) if p.is_file()}
    Path('reports/artifact_manifest.json').unlink(missing_ok=True)
    manifest.pop('reports/artifact_manifest.json', None)
    Path('reports/artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cells = []
    cells.append(nbf.v4.new_markdown_cell(TITLE_TEXT))
    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
from IPython.display import display, Image
root = Path.cwd()
manifest = json.loads(Path('reports/artifact_manifest.json').read_text())
for file, expected in manifest.items():
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == expected, file
result = json.loads(Path('reports/hasil.json').read_text())
display(result)
"""))
    for title, description, table in SECTIONS:
        cells.append(nbf.v4.new_markdown_cell('## ' + title + '\n\n' + description))
        cells.append(nbf.v4.new_code_cell(f"table = pd.read_csv('reports/{table}')\ndisplay(table)"))
    cells.append(nbf.v4.new_markdown_cell('## Pemeriksaan konsistensi\n\nSaya menghitung ulang hubungan pokok dari prediksi tersimpan.'))
    cells.append(nbf.v4.new_code_cell(VERIFY_CODE))
    for figure in FIGURES:
        cells.append(nbf.v4.new_code_cell(f"display(Image(filename='assets/{figure}'))"))
    cells.append(nbf.v4.new_markdown_cell('## Interpretasi dan batas penggunaan\n\nSaya membaca hasil sebagai studi portofolio. Penjelasan lengkap, audit data, keputusan berbasis latih, dan keterbatasan tersedia di reports/laporan_analisis.md.'))
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    NotebookClient(notebook, timeout=300, resources={'metadata': {'path': str(root)}}).execute(env=env)
    Path('notebooks').mkdir(exist_ok=True)
    nbf.write(notebook, 'notebooks/analisis.ipynb')
    print('Notebook dieksekusi dan seluruh hash artefak cocok.')



TITLE_TEXT = '# Credit Risk Scorecard\n\nSaya meninjau artefak hasil eksekusi pipeline. Notebook ini memverifikasi hash sumber tabel/grafik dan menghitung ulang metrik penting tanpa melatih ulang pada data uji.'
SECTIONS = [('Kualitas dan ringkasan data', 'Saya memeriksa ukuran dan distribusi atribut sebelum menafsirkan model.', 'eda.csv'), ('IV pada data latih', 'Saya memilih atribut dari IV latih saja; setiap fold menghitung ulang bin, WoE, dan IV.', 'information_value.csv'), ('Koefisien logistik', 'Saya menafsirkan koefisien dengan target bad=1. P-value pascaseleksi bersifat eksploratif.', 'coefficients.csv'), ('Evaluasi uji', 'Saya menghitung metrik pada 200 baris uji sesudah membekukan model dan cutoff.', 'evaluation.csv'), ('Cutoff dan biaya', 'Saya memilih cutoff memakai OOF latih; tabel uji berikut hanya sensitivitas.', 'cutoffs.csv'), ('Audit kelompok', 'Saya melaporkan ukuran kelompok dan interval approval, tanpa menyimpulkan kepatuhan.', 'fairness.csv')]
FIGURES = ['evaluation.png', 'information_value.png']
VERIFY_CODE = "from sklearn.metrics import roc_auc_score, brier_score_loss\nfrom src.scorecard import score_to_probability, decision_cost\npred = pd.read_csv('reports/test_predictions.csv')\nassert np.allclose(score_to_probability(pred.score), pred.probability)\nassert np.isclose(roc_auc_score(pred.default, pred.probability), result['scorecard']['auc'])\nassert np.isclose(brier_score_loss(pred.default, pred.probability), result['scorecard']['brier'])\nassert decision_cost(pred.default, pred.probability, result['test_decision']['cutoff'])['cost'] == result['test_decision']['cost']\nprint('Skor, probabilitas, AUC, Brier, dan biaya konsisten.')"

if __name__ == '__main__':
    main()
