# Credit Risk Scorecard

Saya membangun scorecard risiko kredit berbasis WoE dan regresi logistik pada data German Credit.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![CI](https://github.com/imman20/credit-risk-scorecard/actions/workflows/ci.yml/badge.svg)
![MIT](https://img.shields.io/badge/License-MIT-green)

![Hasil evaluasi](assets/evaluation.png)

## Ringkasan untuk recruiter

Saya mengolah 1000 catatan kredit menjadi pipeline scorecard yang memisahkan pembelajaran transformasi dari evaluasi uji.
Saya membandingkan logistik WoE dan gradient boosting, serta menghubungkan prediksi dengan poin dan biaya keputusan.
Saya menyediakan notebook yang dieksekusi, laporan analisis, pengujian, dan aplikasi pendidikan Streamlit.

## Latar belakang dan pertanyaan

Saya menilai kemampuan scorecard membedakan risiko, mengkalibrasi probabilitas, dan memilih
cutoff saat biaya salah setuju lima kali salah tolak. Saya juga memeriksa stabilitas dan
perbedaan hasil antarkelompok tanpa mengklaim kelayakan operasional.

## Data

[UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data),
1.000 baris dan 20 atribut. Target sumber 1=good, 2=bad dipetakan ke default 0/1.
Sumber diunduh melalui `src/data/download.py`; URL dan SHA256 di `reports/source.json`.
Data penuh tidak dikomit; sampel 12 baris ada di `data/sample/`. Lisensi data CC BY 4.0,
sitasi: Hofmann, H. (1994), DOI 10.24432/C5NC77. [Kamus atribut](docs/data_dictionary.md).

## Metodologi

```mermaid
flowchart LR
  A[UCI dan validasi] --> B[Split stratified]
  B --> C[Latih: CV WoE dan seleksi IV]
  C --> D[Logistik dan cutoff OOF]
  B --> E[Latih: CV boosting]
  D --> F[Evaluasi uji sekali]
  E --> F
  F --> G[Poin, fairness, PSI dan aplikasi]
```

Saya membagi 80:20 secara stratified dengan seed 2024 sebelum transformasi. Saya memakai
empat bin kuantil numerik, kategori asli untuk atribut kategorik, smoothing 0,5 per kelas/bin,
dan WoE = ln(distribusi good/distribusi bad). Saya memilih IV >= 0,02 pada data latih;
ambang ini merupakan aturan eksploratif, bukan jaminan stabilitas. Bila tak ada yang lolos,
fit memilih satu IV tertinggi. Median, bin, WoE, dan seleksi IV dipelajari ulang di setiap
fold latih pada 5-fold stratified CV. Boosting memakai imputasi dan one-hot yang juga
di-fit ulang per fold. Hyperparameter ditetapkan sebelum evaluasi uji.

Saya mengestimasi logistik bad=1 melalui GLM Binomial statsmodels tanpa regularisasi.
Koefisien positif menaikkan log-odds bad per satu unit WoE dengan atribut lain tetap.
P-value dan interval Wald bersifat eksploratif setelah seleksi IV, bukan inferensi konfirmatori.
Konvergensi diwajibkan oleh pipeline; separation atau kegagalan fit menghentikan pipeline,
bukan disembunyikan dengan estimator pengganti. Korelasi antaratribut dan ukuran data
membatasi interpretasi individual.

Saya menetapkan faktor = 20/ln(2) dan offset = 600 - faktor*ln(50).
Skor = offset - faktor*logit(p_bad), sehingga p_bad = 1/(1+exp((skor-offset)/faktor)).
Skor 600 berarti odds good:bad 50:1. Kenaikan 20 poin menggandakan odds good:bad.
Base points = offset - faktor*intersep; poin tiap bin = -faktor*koefisien*WoE.
Kategori baru mendapat WoE 0 dan kontribusi poin 0; base points tetap ditambahkan.
Saya memverifikasi penjumlahan poin terhadap probabilitas tanpa pembulatan. Pembulatan
poin hanya boleh untuk tampilan, bukan pengambilan keputusan.

Saya menolak bila p_bad >= cutoff. Salah setuju adalah bad yang disetujui (false negative)
dengan biaya 5; salah tolak adalah good yang ditolak (false positive) dengan biaya 1.
Cutoff dipilih dari prediksi out-of-fold data latih untuk meminimumkan 5*FN+FP;
bila seri saya memilih tingkat persetujuan tertinggi. Evaluasi data uji dilakukan setelah
keputusan ini dibekukan. Tabel cutoff lain hanya sensitivitas, tidak dipakai memilih ulang.
CV ini mengevaluasi prosedur belajar, sementara ketidakpastian pemilihan cutoff belum
diukur lewat nested CV. Biaya adalah unit relatif, bukan nominal uang atau estimasi laba.


## Hasil utama

| Metrik uji | Scorecard | Gradient boosting |
|---|---:|---:|
| AUC | 0.765119 | 0.777262 |
| Gini | 0.530238 | 0.554524 |
| KS | 0.476190 | 0.461905 |
| Brier | 0.175006 | 0.169507 |

Saya memperoleh cutoff OOF 0.169345. Pada 200 baris uji,
biaya relatif = 113, salah setuju = 8, salah tolak = 73,
dan tingkat persetujuan = 37.50%. PSI probabilitas latih versus uji =
0.083813. AUC OOF scorecard = 0.774814,
boosting = 0.762827. Base points = 511.986462.
Error maksimum konsistensi poin = 1.14e-13.


[Laporan lengkap](reports/laporan_analisis.md), [tabel IV](reports/information_value.csv),
[poin](reports/scorecard_points.csv), dan [fairness](reports/fairness.csv).

## Struktur folder

```text
src/         data/download.py, scorecard.py, pipeline.py, reporting.py, notebook.py
app/         streamlit_app.py
tests/       unit, leakage, smoke pipeline, dan aplikasi
notebooks/   analisis.ipynb dengan output eksekusi
reports/     hasil.json, laporan dan tabel CSV
assets/      gambar hasil eksekusi
data/sample/ sampel kecil data sumber
```

## Cara menjalankan

Gunakan Python 3.11 dari root repo.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
python -m ruff check .
python -m pytest -q
streamlit run app/streamlit_app.py
```

`python run_all.py` mengunduh data, menjalankan analisis, dan mengeksekusi notebook.
Artefak lokal `artifacts/scorecard.joblib` dibangun pipeline dan tidak dikomit.
Aplikasi menampilkan pesan petunjuk bila artefak belum tersedia. Kategori di luar data
latih menggunakan WoE netral dan diberi peringatan. CI memakai fixture kecil tanpa unduhan.

## Keterbatasan

Saya menggunakan data kredit historis Jerman berukuran kecil. Label good/bad adalah
penilaian kredit sumber dan tidak mendefinisikan horizon default operasional. Hasil split
tunggal dapat berubah pada seed lain. Saya tidak menyimpulkan hubungan sebab akibat.
PSI memakai bin desil dari latih dan floor proporsi 1e-6; split acak bukan bukti pergeseran
populasi antarwaktu. Monitoring operasional memerlukan periode baru serta label yang matang.

Saya memetakan A91/A93/A94 ke male dan A92/A95 ke female sesuai kamus sumber.
Personal status/sex adalah atribut gabungan, bukan pengukuran gender modern atau lengkap.
Saya mempertahankan atribut ini untuk audit demonstrasi dan mengakui pemakaiannya dalam
model dapat memengaruhi keputusan. Kelompok umur ditetapkan <25, 25-49, dan 50+ sebelum
melihat hasil. Tabel fairness memuat ukuran, jumlah bad, approval dengan interval Wilson 95%,
false reject pada good, false approve pada bad, serta Brier. Kelompok kecil menghasilkan
ketidakpastian besar; perbedaan bukan bukti diskriminasi kausal atau jaminan keadilan.
Saya tidak membuat klaim kepatuhan. Aplikasi adalah demonstrasi pendidikan dan bukan
alat persetujuan kredit operasional.


## Lisensi

Kode berlisensi [MIT](LICENSE) atas nama Immantaka Juniarsa; lisensi data tetap mengikuti UCI.

## Kontak

Immantaka Juniarsa, S1 Statistika Universitas Airlangga (2024).
[GitHub imman20](https://github.com/imman20), immantakajuniarsa30@gmail.com.
