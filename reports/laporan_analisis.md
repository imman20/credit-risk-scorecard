# Laporan Credit Risk Scorecard

Saya membangun scorecard yang dapat direproduksi dan membandingkannya dengan gradient boosting.
Seluruh angka bersumber dari eksekusi `python -m src.pipeline`, terutama `reports/hasil.json`.

## Data dan kualitas

Saya memakai 1000 baris UCI, 300 label bad, 800 latih dan 200 uji.
Sel kosong = 0; duplikasi persis = 0.
Saya mempertahankan duplikasi karena tidak ada ID untuk memastikan peminjam yang sama.
Kamus dan sumber: [docs/data_dictionary.md](../docs/data_dictionary.md), hash di `source.json`.

## Metode dan keputusan

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


## Hasil

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


![Evaluasi](../assets/evaluation.png)

## IV dan alasan pemilihan

| attribute           |          iv | selected   |
|:--------------------|------------:|:-----------|
| checking            | 0.602257    | True       |
| history             | 0.284611    | True       |
| purpose             | 0.191645    | True       |
| savings             | 0.190332    | True       |
| duration            | 0.160005    | True       |
| amount              | 0.146464    | True       |
| age                 | 0.119345    | True       |
| employment          | 0.101306    | True       |
| personal_status_sex | 0.0925594   | True       |
| property            | 0.0811803   | True       |
| housing             | 0.0809159   | True       |
| other_plans         | 0.0576279   | True       |
| existing_credits    | 0.0335227   | True       |
| debtors             | 0.028067    | True       |
| foreign_worker      | 0.0193154   | False      |
| installment         | 0.0159692   | False      |
| job                 | 0.0146226   | False      |
| telephone           | 0.00928489  | False      |
| residence           | 0.00386722  | False      |
| dependents          | 0.000301914 | False      |

## Koefisien dan ketidakpastian

|                     |   coefficient |     p_value |   ci_low |    ci_high |
|:--------------------|--------------:|------------:|---------:|-----------:|
| const               |     -0.861706 | 6.10951e-20 | -1.04644 | -0.676972  |
| checking            |     -0.824633 | 2.95715e-11 | -1.06773 | -0.58154   |
| duration            |     -0.856675 | 0.000437409 | -1.33416 | -0.379186  |
| history             |     -0.627506 | 0.00104284  | -1.00262 | -0.252392  |
| purpose             |     -1.00647  | 4.31855e-06 | -1.43573 | -0.577206  |
| amount              |     -0.947913 | 0.000131931 | -1.43391 | -0.461918  |
| savings             |     -0.814825 | 0.000350758 | -1.2616  | -0.368055  |
| employment          |     -0.622552 | 0.0371785   | -1.20811 | -0.0369966 |
| personal_status_sex |     -0.927879 | 0.00314233  | -1.54363 | -0.312123  |
| debtors             |     -1.32419  | 0.0139255   | -2.37957 | -0.268812  |
| property            |     -0.312489 | 0.398768    | -1.03831 |  0.413335  |
| age                 |     -0.645009 | 0.0225915   | -1.19941 | -0.0906041 |
| other_plans         |     -0.909095 | 0.0161478   | -1.64979 | -0.168397  |
| housing             |     -0.568368 | 0.0970569   | -1.23973 |  0.102991  |
| existing_credits    |     -0.170368 | 0.763282    | -1.27908 |  0.938346  |

Konvergen: True. Peringatan fit: tidak ada.
Tabel poin lengkap: `scorecard_points.csv`; tabel bin dan distribusi: `woe.csv`.

## Sensitivitas cutoff dan confusion matrix

|   cutoff |   cost |   wrong_approve |   wrong_reject |   tn |   fp |   fn |   tp |   approval_rate |
|---------:|-------:|----------------:|---------------:|-----:|-----:|-----:|-----:|----------------:|
| 0.1      |    120 |               5 |             95 |   45 |   95 |    5 |   55 |           0.25  |
| 0.169345 |    113 |               8 |             73 |   67 |   73 |    8 |   52 |           0.375 |
| 0.2      |    102 |               8 |             62 |   78 |   62 |    8 |   52 |           0.43  |
| 0.3      |    115 |              14 |             45 |   95 |   45 |   14 |   46 |           0.545 |
| 0.5      |    159 |              27 |             24 |  116 |   24 |   27 |   33 |           0.715 |

TN = jumlah good uji - salah tolak; TP = jumlah bad uji - salah setuju.
Seluruh label dan prediksi tersedia pada `test_predictions.csv` untuk rekonstruksi.

## Fairness dan monitoring

| attribute   | group   |   n |   bad_n |   approval_rate |   approval_ci_low |   approval_ci_high |   false_reject_rate_good |   false_approve_rate_bad |    brier |
|:------------|:--------|----:|--------:|----------------:|------------------:|-------------------:|-------------------------:|-------------------------:|---------:|
| age         | 25-49   | 145 |      43 |        0.386207 |          0.310869 |           0.467418 |                 0.5      |                 0.116279 | 0.157016 |
| age         | 50+     |  23 |       7 |        0.478261 |          0.292369 |           0.670376 |                 0.4375   |                 0.285714 | 0.195485 |
| age         | <25     |  32 |      10 |        0.25     |          0.132522 |           0.421069 |                 0.681818 |                 0.1      | 0.241805 |
| sex         | female  |  64 |      15 |        0.296875 |          0.19905  |           0.417704 |                 0.612245 |                 0        | 0.147838 |
| sex         | male    | 136 |      45 |        0.411765 |          0.332581 |           0.495796 |                 0.472527 |                 0.177778 | 0.187791 |

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
