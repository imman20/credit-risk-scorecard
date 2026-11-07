# Sumber, label, dan kamus atribut

Saya menggunakan Statlog (German Credit Data), Hofmann (1994), UCI, DOI
https://doi.org/10.24432/C5NC77. Arsip sumber:
https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip.
Lisensi data CC BY 4.0. Saya menyimpan URL dan SHA256 arsip di `reports/source.json`.
`src/data/download.py` mengunduh `german.data` dan `german.doc`, lalu memvalidasi
1.000 baris, 20 atribut, serta target. `german.doc` asli tetap tersedia sesudah unduhan.

Target sumber `1=good`, `2=bad`. Saya memetakan `default = (label == 2)` sehingga
bad=1 adalah kelas positif. Istilah probabilitas default di aplikasi merujuk pada label
bad historis ini, bukan kejadian default dalam horizon waktu yang telah diverifikasi.
Matriks biaya sumber juga memberi bobot lima untuk salah menganggap bad sebagai good.

| Nama paket | Makna dan unit/kode sumber |
|---|---|
| checking | A11 saldo giro <0 DM; A12 0 sampai <200 DM; A13 >=200 DM atau gaji >=1 tahun; A14 tanpa rekening |
| duration | Durasi kredit dalam bulan |
| history | A30 tanpa kredit/semua lunas; A31 kredit bank ini lunas; A32 kredit berjalan dibayar lancar; A33 keterlambatan historis; A34 akun kritis/kredit lain |
| purpose | A40 mobil baru; A41 mobil bekas; A42 perabot; A43 radio/TV; A44 alat rumah tangga; A45 perbaikan; A46 pendidikan; A48 pelatihan ulang; A49 bisnis; A410 lainnya |
| amount | Jumlah kredit dalam Deutsche Mark historis, bukan rupiah |
| savings | A61 <100 DM; A62 100 sampai <500; A63 500 sampai <1.000; A64 >=1.000; A65 tidak diketahui/tanpa tabungan |
| employment | A71 tidak bekerja; A72 <1 tahun; A73 1 sampai <4; A74 4 sampai <7; A75 >=7 |
| installment | Tingkat cicilan terhadap pendapatan disposabel, kode integer sumber 1-4 |
| personal_status_sex | A91 laki-laki cerai/pisah; A92 perempuan cerai/pisah/menikah; A93 laki-laki lajang; A94 laki-laki menikah/duda; A95 perempuan lajang |
| debtors | A101 tidak ada; A102 co-applicant; A103 penjamin |
| residence | Lama tinggal saat ini, kode integer sumber 1-4 |
| property | A121 real estat; A122 tabungan perumahan/asuransi jiwa; A123 mobil/lainnya; A124 tidak diketahui/tidak ada |
| age | Umur dalam tahun |
| other_plans | A141 bank; A142 toko; A143 tidak ada |
| housing | A151 sewa; A152 milik sendiri; A153 gratis |
| existing_credits | Jumlah kredit berjalan pada bank ini |
| job | A171 tidak bekerja/tidak terampil nonresiden; A172 tidak terampil residen; A173 terampil/pegawai; A174 manajemen/wiraswasta/berkualifikasi tinggi |
| dependents | Jumlah orang yang ditanggung |
| telephone | A191 tidak ada; A192 ada atas nama peminjam |
| foreign_worker | A201 ya; A202 tidak |

Saya menggunakan kode kategori asli agar pemetaan ke sumber dapat ditelusuri.
Nilai unknown yang memang kategori sumber, seperti A65, tetap kategori tersendiri.
Kategori yang belum pernah terlihat saat fit berbeda dari kategori unknown sumber:
transformer memberi WoE netral 0 tanpa mengubah parameter yang telah dipelajari.
Imputasi numerik memakai median latih; nilai hilang kategorik memakai token khusus.
Aplikasi meminta seluruh atribut dan menolak nilai numerik tidak positif/tidak hingga
serta umur di luar 18-120. Masukan kategori baru diterima dengan peringatan.

Saya tidak memiliki ID peminjam, tanggal aplikasi, horizon label, atau rincian
proses pemberian kredit. Atribut sex/status bersifat gabungan dan membatasi audit
kelompok. Saya tidak menafsirkan kode tersebut sebagai identitas gender lengkap.
