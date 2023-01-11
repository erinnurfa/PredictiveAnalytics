# Laporan Proyek Machine Learning - Erin Nur Fatimah

## Domain Proyek
Susu sapi mengandung protein hewani yang sangat besar manfaatnya bagi bayi maupun mereka yang sedang dalam proses pertumbuhan, karena susu sapi mengandung asam amino esensial dalam jumlah yang cukup. UTP Laboratorium Kesehatan Hewan Malang sebagai unit pelaksana teknis di bawah Dinas Peternakan Jawa Timur bertugas melakukan pengujian di bidang kesmavet untuk upaya pengamanan susu sebagai produk peternakan dengan pengujian yang tepat sesuai dengan Standar Nasional Indonesia (SNI). Pengklasifikasian kualitas susu sapi di UPT tersebut masih dilakukan secara organoleptic (bau, rasa, dan warna) yang bersifat linguistik sehingga variabel dan penentuan parameter bersifat tidak pasti dan menjadi kendala utama pakar dalam menentukan kualitas susu yang baik. [[1](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/2935)].
Untuk itu dalam proyek ini saya mengangkat judul **Prediksi Kualitas susu**. Dengan adanya model *machine learning* ini diharapkan dapat memudahkan pekerjaan para pakar susu dalam mengidentifikasi kualitas susu dengan baik.

## Business Understanding
Berdasarkan latar belakang diatas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini:
### Problem Statements
* Bagaimana membuat model machine learning untuk memprediksi kualitas susu dengan baik?
* Bagaimana membuat algoritma yang mampu menghasilkan akurasi paling baik?
### Goals
* Membuat model machine learning untuk memprediksi kualitas susu dengan baik.
* Membuat model machine learning dengan nilai akurasi yang minimal mencapai 85%.
##### Solution Statements:
* Untuk mencapai tujuan dari proyek ini kita akan menggunakan 4 algoritma yaitu DecisionTreesClassifier, KNeighborsClassifier, Random Forest, dan Gradien Boosting.
* Menerapkan metrik evaluasi pada model Machine Learning yang telah dibuat untuk mengetahui kualitas model terkait.

## Data Understanding
Dataset yang digunakan bersumber dari platform penyedia dataset yaitu Kaggle. Dataset ini memiliki 8 kolom atau dimensi dan total 1059 baris atau objek. Berikut dataset penulis gunakan :
**Jenis**|**Keterangan**|
:-----:|:-----:
Sumber| [Kaggle dataset : Milk Quality Prediction](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality)|
Jenis dan Ukuran Berkas | CSV(1 kB)

Data yang digunakan memiliki fitur-fitur berikut ini :
* pH : kadar pH dari susu
* Temperature : suhu dari susu
* Taste : rasa dari susu
* Odor : bau dari susu
* Fat : kadar lemak dari susu
* Turbidity : kadar kekeruhan dari susu
* Colour : warna darisusu
* Grade : kualitas dari susu

Dari 8 fitur yang ada kita akan berfokus pada fitur grade, berbasis pada fitur ini kita akan mencari tahu mengenai tingkat kualitas dari susu dengan nilai “low”, ”medium”, dan “hight”.

Pada tahapan data ini kita akan mengetahui berbagai macam informasi dataset yang dipakai, diantaranya sebagai berikut :
1.	Langkah pertama yang perlu dilakukan adalah mengimport beberapa library pendukung, library ini ada beberapa macam, seperti library untuk model, untuk handle data dan untuk visualisasi.
2.	Selanjutnya untuk mengimport dataset agar bisa dilakukan preparation, bisa menggunakan ‘’’pd.read_csv’’’ dari pandas.
3.	Melihat informasi data (kolom dan jumlah data) dengan menggunakan fungsi shape.
4.	Melihat data yang memiliki missing value pada semua fitur yang ada. Untuk melakukan ini kita akan menggunakan fungsi berikut ini dimana ‘’’ds’’’ adalah variabel untuk dataset saat kita import sebelumnya.
‘’’ds.isna().sum()

## Data Preparation
Sebelum menjalankan tahap persiapan data, maka kita perlu melakukan beberapa langkah.
1.	```df.info()``` digunakan untuk mengecek tipe kolom pada dataset
2.	```df.describe()``` digunakan utk mendapatkan info mengenai dataset terhadap nilai rata-rata, median, banyaknya data, nilai Q1 hingga Q3 dan lain-lain.
3.	```df.isna().sum()``` digunakan untuk mengecek apakah ada kolom yg kosong, pada dataset ini nilai kosong tidak ditemukan
4.	```df.shape()``` digunakan untuk melihat hasil akhir data 
5.	Menangani Outlier
Beberapa pengamatan dalam satu set data kadang berada di luar lingkungan pengamatan lainnya. Pengamatan seperti itu disebut outlier. Menurut Kuhn dan Johnson dalam Applied Predictive Modeling, outliers adalah sampel yang nilainya sangat jauh dari cakupan umum data utama. Ia adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya. Jika terdapat fitur pada dataset yang memiliki outlier kita bisa menggunakan IQR Method. Untuk melihat outlier kita bisa memanfaatkan ```boxplot```. Jika terdapat fitur pada dataset yang memiliki outlier kita bisa mengatasinta menggunakan IQR Method.
6.	Mengatasi outlier dengan IQR Method
Data yang berada di luar Q1 dan Q3 adalah outlier, dimana kita akan menentukan nilai batas atas dan bawah, dengan persamaan berikut :
Batas bawah = Q1 – 1.5 * IQR
Batas atas = Q3 + 1.5 * IQR
Setelah outlier data dipangkas maka yang sebelumnya 1059 data  menjadi 648 data.
7.	Membagi fitur berdasarkan kategori tipe datanya.
- Object : akan dianggap fitur categorical
- Selain object (Float, Int) : akan dianggap fitur numerical

Berikut beberapa tahapan visualisasi data :
1. Melakukan visualisasi data dalam bentuk sns.countplot.Penggunaan sns.countplot sendiri berfungsi untuk menghitung jumlah data yang sama.
![Gambar 1](https://i.postimg.cc/LX7s9jPz/gambar0.png ) 
Bisa kita lihat pada beberapa variable plot diatas bahwa kualitas medium atau kelas menengah cukup tinggi dibandingkan dengan yang kualitas low ataupun yang hight.
2. Melakukan visualisasi distribusi numerik, bisa dapat kita lihat pada gambar berikut,
![Gambar 2](https://i.postimg.cc/YS2k5mQ3/gambar2.png)
Fungsi pairplot dari library seaborn menunjukkan relasi pasangan dalam dataset.  Dari gambar grafik, kita dapat melihat plot relasi masing-masing dataset.
3. Selanjutnya visualisasi dilakukan untuk mengetahui korelasi antar fitur yang terdapat pada dataset, bisa kita lihat selengkapnya sebagai berikut
![Gambar 3](https://i.postimg.cc/brxBYmh2/gambar3.png)
Koefisien korelasi berkisar antara -1 dan +1. Ia mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atay negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1 atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah.
Arah korelasi antara dua variabel bisa bernilai positif (nilai kedua variabel cenderung meningkat bersama-sama) maupun negatif (nilai salah satu variabel cenderung meningkat ketika nilai variabel lainnya menurun).
Pada gambar di atas. Jika kita amati , korelasi pada fitur-fitur dataset mem


Dengan dibuatnya grafik-grafik di atas, kita sudah mendapatkan banyak data yang memudahkan kita untuk proses analisis data.

Untuk persiapan data, kita bisa menggunakan beberapa teknik yang diperlukan dalam tahap persiapan data. Sebagai berikut:
- Train Test Split : Proporsi pembagian data latih dan uji adalah 80:20. Tujuan dari data uji adalah untuk mengukur kinerja model pada data baru. Pembagian dataset ini menggunakan modul train_test_split dari scikit-learn.
- Standarisasi : Langkah terakhir adalah standarisasi data. Proses standardisasi mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1. Untuk merangkai data ini menggunakan fungsi StandardScaler. Kita akan menggunakan StandarScaler dari library Scikitlearn.

## Modeling
Setelah melakukan preprocessing dataset, langkah selanjutnya adalah memodelkan data. Empat algoritma digunakan pada tahap ini: Gradient Boosting, Decision Tree, K-Nearest Neighbor, dan Random Forest. 

**Gradient Boosting**
Gradient boosting adalah algoritma machine learning yang menggunakan ensambel dari decision tree untuk memprediksi nilai. 
`[+]` Kelebihan :
-	Nilai akurasinya lebih tinggi jika dibandingkan dengan algoritma yang lain
-	Mampu menangani complex pattern dan data ketika linear model tidak dapat menangani hubungan antar fitur dalam dataset

`[-]` Kekurangan :
-	Mempunyai biaya komputasi yang mahal
-	Mudah mengalami overfifting karena algoritma ini sensitive terhadap outlier

```
mod1 = GradientBoostingClassifier(random_state=123)
prm1 = {
    'classifier__n_estimators':[10,50,100,250],
    'classifier__max_depth':[5,10,20],
    'classifier':[mod1]
}
```

**Decision Tree**
Decision Tree atau Pohon keputusan merupakan suatu metode klasifikasi yang paling populer karena mudah untuk diinterpretasikan oleh manusia, yang berbentuk struktur pohon untuk mengambil keputusan tersebut.
Konsep dari decision tree ini yaitu dengan mengubah data menjadi decision tree dan aturan-aturan keputusan. Sedangkan manfaat utama dari pohon keputusan ini yaitu kemampuan untuk menyederhanakan proses pengambilan keputusan yang kompleks sehingga pembuat keputusan dapat menafsirkan solusi untuk menyelesaikan masalah.
`[+]` Kelebihan :
- Mudah dibaca dan ditafsirkan
- Lebih sedikit pembersihan data yang diperlukan
- Integrasi yang mudah ke dalam sistem basis data
- Memiliki akurasi yang baik
- Lokasi keputusan yang sebelumnya sangat kompleks dan sangat global dapat dibuat dengan lebih sederhana dan lebih spesifik.
- Dapat menghilangkan perhitungan yang tidak diperlukan. Dikarenakan dengan metode ini sampel hanya akan diuji berdasarkan kriteria ataupun kelas tertentu.
- Terdapat pemilihan fitur yang fleksibel dari node internal yang berbeda, sehingga fitur yang dipilih membedakan kriteria lain di node yang sama.

`[-]` Kekurangan :
- Tumpang tindih yang terjadi terutama ketika sangat banyak kelas dan kriteria yang digunakan. Ini dapat menyebabkan waktu keputusan yang lebih lama dan memori yang dibutuhkan lebih banyak.
- Perhitungan jumlah kesalahan dari setiap level dalam pohon keputusan lumayan besar.
- Kesulitan dalam merancang pohon keputusan yang optimal
- Hasil kualitas keputusan yang didapat dengan menggunakan metode pohon keputusan sangat tergantung kepada bagaimana pohon itu dirancang.

```
mod2 = DecisionTreeClassifier(random_state=123)
prm2 = {
    'classifier__max_depth':[5,10,25],
    'classifier__min_samples_split':[2,5,10],
    'classifier':[mod2]
}
```

**K-Nearest Neighbor**
KNN adalah algoritma relative sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan K-nearest neighboar (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi. Pada modul ini, kita akan menggunakannya untuk kasus klasifikasi.
`[+]` Kelebihan :
- Mudah diterapkan
- Mudah beradaptasi
- Memiliki sedikit hyperparameter

`[-]` Kekurangan :
- Tidak berfungsi dengan baik pada dataset berukuran besar
- Kurang cocok untuk dimensi tinggi
- Perlu penskalaan fitur
- Sensitif terhadap noise data, missing values dan outliers

```
mod3 = KNeighborsClassifier()
prm3 = {
    'classifier__n_neighbors':[2,5,10,25,50,100],
    'classifier':[mod3]
}
```

**Random Forest**
Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Ensembel merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir.
`[+]` Kelebihan :
- Kuat terhadap data outlier (pencilan data).
- Bekerja dengan baik dengan data non-linear.
- Risiko overfitting lebih rendah.
- Berjalan secara efisien pada kumpulan data yang besar.
- Akurasi yang lebih baik daripada algoritma klasifikasi lainnya.

`[-]` Kekurangan :
- Random Forest cenderung bias saat berhadapan dengan variabel kategorikal.
- Waktu komputasi pada dataset berskala besar relatif lambat
- Tidak cocok untuk metode linier dengan banyak fitur sparse

```
mod4 = RandomForestClassifier(random_state=123)
prm4 = {
    'classifier__n_estimators':[10,50,100,250],
    'classifier__max_depth':[5,10,20],
    'classifier':[mod4]
}
```
kemudian kita akan melakukan training pada model - model & hyperparameter tuning dengan fungsi ```gridsearchcv()```. Setelah itu kita menampilkan overview data dari laporan hasil training & hyperparameter tuning terhadap beberapa model. Pada model dengan algoritma Gradient boosting memiliki nilai akurasi lebih tinggi dibanding dengan algoritma Decision Tree, KNN, dan Random Forest. 

## Evaluation
Proyek ini menggunakan empat metrik. Empat metrik tersebut adalah:
* Accuracy : merupakan rasio prediksi benar (positif dan negatif) dengan keseluruhan data. Rumus Accuracy Score = (TP + TN)/ (TP + FN + TN + FP).
* Precision : merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positif. Rumus Precision Score = TP / (FP + TP)
* Recall (Sensifitas) : merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif.
Rumus Recall Score = TP / (FN + TP)
* F1 Score : merupakan perbandingan rata-rata presisi dan recall yang dibobotkan. Rumus F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)

Keterangan istilah dari Confision Matrix :
True Negative (TN) : model memprediksi data ada di kelas negative dan yang sebenarnya data memang di kelas negative
True Positive (TP) : model memprediksi data ada di kelas positif dan yang sebenarnya data memang ada di kelas positif.
False Negative (FN) : model memprediksi data ada di kelas negative, namun yang sebenarnya data ada di eklas positif.
False Positive (FP) : model memprediksi data ada di kelas positif, namun yang sebenarnya data ada di kelas negative.

Berikut adalah hasil evaluasi dari keempat matriks dari model Gradient boosting.
![Gambar 4](https://i.postimg.cc/HjDj223x/gambar4.png)
**Kesimpulan**
Pada model algoritma gradient boosting berada di urutan pertama dengan validation skor mencapai akurasi sebesar 0.9923 atau 99%. Penyebab model algoritma gradient boosting menjadi model yang paling baik dikarenakan model ini sudah terbukti unggul jika dibandingkan dengan algoritma-algotitma yang lain. Selain itu algoritma ini juga mampu menangani complex pattern dan data ketika linear model tidak dapat menangani hubungan antar fitur dalam dataset.

### Referensi 
[1] Kurnia, A., Furqon, M., & Rahayudi, B. Klasifikasi Kualitas Susu Sapi Menggunakan Algoritme Support Vector Machine (SVM) (Studi Kasus: Perbandingan Fungsi Kernel Linier dan RBF Gaussian). Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer, vol. 2, no. 11, p. 4453-4461, mar. 2018. 
