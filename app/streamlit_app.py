"""Formulir pendidikan berbasis artefak hasil pipeline lokal."""
import sys
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.download import COLUMNS, NUMERIC  # noqa: E402
from src.scorecard import probability_to_score, validate_input  # noqa: E402
from src.deployment import PortableScorecard  # noqa: E402

st.set_page_config(page_title='Credit Risk Scorecard', layout='wide')
st.title('Credit Risk Scorecard')
st.warning('Demonstrasi pendidikan, bukan alat persetujuan kredit operasional.')
path = ROOT/'app/scorecard.json'
if not path.exists():
    st.info('Parameter belum tersedia. Jalankan python run_all.py dari root repo.')
    st.stop()


@st.cache_resource
def load_artifact(stamp):
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {**payload, 'model': PortableScorecard(payload)}


bundle = load_artifact(path.stat().st_mtime)
st.caption('Target bad sesuai label historis UCI. Kode atribut dijelaskan di docs/data_dictionary.md.')
st.caption(f"Disetujui bila probabilitas bad < {bundle['cutoff']:.6f}; cutoff dipilih dari prediksi OOF latih.")
with st.form('applicant'):
    values = {}
    columns = st.columns(3)
    for i, col in enumerate(COLUMNS):
        with columns[i % 3]:
            if col in NUMERIC:
                values[col] = st.number_input(col, min_value=18 if col == 'age' else 1,
                                              max_value=120 if col == 'age' else None,
                                              value=int(bundle['defaults'][col]))
            else:
                values[col] = st.text_input(col, value=str(bundle['defaults'][col]),
                                            help='Kategori latih: ' + ', '.join(bundle['categories'][col]))
    submitted = st.form_submit_button('Hitung skor')
if submitted:
    try:
        frame = validate_input(pd.DataFrame([values]))
        unknown = [c for c, cats in bundle['categories'].items() if values[c] not in cats]
        if unknown:
            st.warning('Kategori tidak dikenal menggunakan WoE netral: ' + ', '.join(unknown))
        p = float(bundle['model'].predict_proba(frame)[0, 1])
        a, b, c = st.columns(3)
        a.metric('Skor', f'{float(probability_to_score(p)):.2f}')
        b.metric('Probabilitas bad', f'{p:.2%}')
        c.metric('Keputusan demonstrasi', 'Setuju' if p < bundle['cutoff'] else 'Tolak')
        st.caption('Skor lebih tinggi berarti risiko lebih rendah. PDO 20; skor 600 pada odds good:bad 50:1.')
    except (ValueError, TypeError) as error:
        st.error(str(error))
