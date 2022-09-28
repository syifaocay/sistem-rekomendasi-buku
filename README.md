### by Ahmad Syifaul Umam
# Laporan Proyek Machine Learning 

## Project Overview- 

Ditengah cepatnya era digital berkembang, dunia bisnis juga terus berinovasi untuk meningkatkan produk mereka. salah satu yang sudah sering kita jumpai dalam perkembangan bisnis adalah adanya sistem rekomendasi. jika kita melihat di online shop banyak sekali produk yang direkomendasikan untuk kita beli. sistem rekomendasi ini sangat dibutuhkan untuk mendatangkan banyak pengunjung untuk lapak online kita bahkan bisa juga meningkatkan penjualan. untuk membuat sistem rekomendasi kita membutuhkan data atau riwayat user yang ingin kita berikan rekomendasi.
maka dari itu saya pun mencoba membuat inovasi dalam bisnis dengan membuat judul **sistem rekomendasi buku menggunakan metode collaborative filltering**

## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, berikut adalah masalah yang akan kita selesaikan:
- bagaimana cara membuat sistem rekomendasi buku menggunakan metode collaborative filltering dengan data yang tersedia?

### Goals
Tujuan dari proyek ini adalah:
- Menghasilkan rekomendasi buku kepada pengguna berdasarkan rating buku yang telah diberikan sebelumnya dengan metode Collaborative Filtering.

### Solution Statement
untuk membantu pengguna mendapatkan sistem rekomendasi kita akan membuatnya dengan metode collaborative filltering, Metode ini akan menghasilkan rekomendasi sejumlah buku yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya. 
Dari data rating pengguna dapat digunakan untuk mengidentifikasi buku-buku yang mirip dan belum pernah diberi rating oleh pengguna untuk direkomendasikan. 


## Data Understanding

link Dataset: https://www.kaggle.com/code/aleemaparakatta/bookrecommendation-collaborativefiltering/data?select=Users.csv

Informasi Dataset:

Jenis | Keterangan
--- | ---
Title | Book Recommendation Dataset
Source | [Kaggle](https://www.kaggle.com/arashnic/book-recommendation-dataset)
Maintainer | [Möbius](https://www.kaggle.com/arashnic)
License | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
Usability | 10.0

Pada Dataset ini terdapat 3 berkas csv diantaranya yaitu `Books.csv` , `Ratings.csv` , dan `Users.csv`

Pada berkas `Books.csv` memuat data-data buku yang terdiri dari 271.360 baris dan memiliki 8 kolom, diantaranya adalah :  

- `ISBN` : berisi kode ISBN dari buku  
- `Book-Title` : berisi judul buku
- `Book-Author` : berisi penulis buku
- `Year-Of-Publication` : tahun terbit buku  
- `Publisher` : penerbit buku  
- `Image-URL-S` : URL menuju gambar buku berukuran kecil
- `Image-URL-M` : URL menuju gambar buku berukuran sedang
- `Image-URL-L` : URL menuju gambar buku berukuran besar

Pada berkas `Ratings.csv` memuat data rating buku yang diberikan oleh pengguna. Data ini memiliki 1.149.780 baris dan memiliki 3 kolom, yaitu :  

 - `User-ID` : berisi ID unik pengguna
 - `ISBN` : berisi kode ISBN buku yang diberi rating oleh pengguna
 - `Book-Rating` : berisi nilai rating yang diberikan oleh pengguna berkisar antara 0-10

tabel 1.Data rating.

|   | User-ID  |ISBN   |  book-rating |
|---|---|---|---|
|  0 | 276725	  | 034545104X  |  0 |
| 1  | 276726	  | 0155061224  |  5 |
|2   | 276727  |  0446520802	 |  0 |
|  3 | 276729		  | 052165615X  | 3  |
|  4 |  276729				 |0521795028   | 6  |
|  ... |...   | ...  |...   |
|1149775|	276704|	1563526298|	9|
|1149776|	276706	|0679447156|	0|
|1149777|	276709	|0515107662|	10|
|1149778|	276721|	0590442449	|10|
|1149779|	276723|	05162443314|	8|

1149780 rows × 3 columns




Pada data `Rating` ini juga ditemukan bahwa `User-ID` berupa ID angka yang berukuran cukup besar. Lalu `ISBN` merupakan string unik identitas buku gabungan angka dan huruf. Kedua nilai ini nantinya perlu dilakukan encoding agar dapat menghasilkan rekomendasi. Data rating ini juga merupakan data utama dalam membuat sistem rekomendasi dengan Collaborative Filtering pada proyek ini.
 

Berikut ini adalah hasil dari jumlah rating buku yang diberikan oleh user.

tabel 2.jumlah rating yang diberikan user.

|  Book-Rating | User-ID  |ISBN   |
|---|---|---|
|0|	716109|	716109|
|1	|1770|	1770|
|2	|2759	|2759|
|3	|5996|	5996|
|4	|8904	|8904|
|5	|50974	|50974|
|6	|36924	|36924|
|7	|76457	|76457|
|8	|103736	|103736|
|9	|67541|	67541|
|10	|78610	|78610|


Pada tabel di atas dapat diketahui bahwa mayoritas user - ada lebih dari 700 ribu yang memberikan rating 0 pada buku sehingga data ini dikatakan tidak seimbang *(imbalance)*. Untuk itu pada data ini nantinya akan dilakukan penanganan agar dapat lebih seimbang.

Pada berkas `Users.csv` memuat data pengguna. Data ini terdiri dari 278.858 baris dan memiliki 3 kolom, yaitu : 

- `User-ID` : berisi ID unik pengguna
- `Location` : berisi data lokasi pengguna
- `Age` : berisi data usia pengguna

## Data Preparation
Teknik yang digunakan dalam penyiapan data *(Data Preparation)* yaitu:
- **Handling Imbalanced Data** : seperti yang sudah kita ketahui diatas bahwa banyak sekali user yang memberikan rating 0 pada buku, hal ini tentunya membuat rating tidak seimbang imbalance) dan bisa mengakibatkan kinerja model menjadi buruk. kita dapat menghilangkan user yang memberikan nilai 0 dengan menghapus data dengan rating 0 menggunakan (drop) agar kinerja model menjadi lebih baik
- **Encoding** : dilakukan untuk menyandikan `User-ID` dan `ISBN` ke dalam indeks integer. Tahapan ini diperlukan karena kedua data tersebut berisi integer yang tidak berurutan (acak) dan gabungan string. Untuk itu perlu diubah ke dalam bentuk indeks.
- **Randomize Dataset** : pengacakan data agar distribusi datanya menjadi random. Pengacakan data bertujuan untuk mengurangi varians dan memastikan bahwa model tetap umum dan *overfit less*. Pengacakan data juga memastikan bahwa data yang digunakan saat validasi merepresentasikan seluruh distribusi data yang ada.
- **Data Standardization** : Pada data rating yang digunakan pada proyek ini berada pada rentang 0 hingga 10. Penerapan standarisasi menjadi rentang 0 hingga 1 dapat mempermudah saat proses training. Hal ini dikarenakan variabel yang diukur pada skala yang berbeda tidak memberikan kontribusi yang sama pada model fitting & fungsi model yang dipelajari dan mungkin berakhir dengan menciptakan bias jika data tidak distandarisasi terlebih dulu.
- **Data Splitting** : dataset dibagi menjadi 2 bagian, yaitu data yang akan digunakan untuk melatih model (sebesar 80%) dan data untuk memvalidasi model (sebesar 20%). Tujuan dari pembagian data uji dan validasi tidak lain adalah untuk proses melatih model serta mengukur kinerja model yang telah didapatkan.
- **class RecommenderNe** : kita membuat class RecommenderNet dengan keras Model class. Kode class RecommenderNet ini terinspirasi dari tutorial dalam situs Keras dengan beberapa adaptasi sesuai kasus yang sedang kita selesaikan.

## Modeling
Di sini, kita membuat class RecommenderNet dengan keras Model class. Kode class RecommenderNet ini terinspirasi dari tutorial dalam situs Keras dengan beberapa adaptasi sesuai kasus yang sedang kita selesaikan. Terapkan kode berikut. Pada tahap ini, model untuk membuat sistem rekomendasi postingan dengan metode collaborative filtering akan dipersiapkan. Model tersebut akan dibuat dengan pustaka Keras dan diberi nama RecommenderNet. Model akan menghitung skor kecocokan antara pengguna dengan postingan melalui teknik embedding. Pertama, model akan melakukan proses embedding terhadap data pengguna dan postingan. Selanjutnya, model akan melakukan operasi perkalian dot product antara embedding pengguna dan postingan. Terakhir, model akan menambahkan bias untuk setiap pengguna atau postingan. Skor kecocokan akan ditentukan dalam skala 0–1 dengan fungsi aktivasi sigmoid.

Beberapa properti yang digunakan dalam kelas RecommenderNet dan menjadi parameter pada layer embedding untuk menghasilkan model diantaranya:
- `num_users` : jumlah data pengguna
- `num_isbn` : jumlah data buku, dihitung berdasarkan ISBN
- `embedding_size` : ukuran atau dimensi yang digunakan dalam embedding pada data user dan buku

Setelah selesai mempersiapkan kelas model, langkah selanjutnya adalah melakukan compile pada model dengan memberikan argumen berupa jumlah pengguna unik pada data tayangan, jumlah postingan unik pada data tayangan, dan ukuran embedding. Metrik yang digunakan untuk mengukur kualitas model adalah metrik RMSE (Root Mean Squared Error).

Model ini menggunakan Binary Crossentropy untuk menghitung **loss function**, Adam (Adaptive Moment Estimation) sebagai **optimizer**, dan root mean squared error (RMSE) sebagai **metrics evaluation**. 

setelah model selesai melakukan pelatihan, langkah selanjutnya yaitu membuat kelas yang berfungsi untuk menampilkan keluaran dalam bentuk bingkai data ataupun teks laporan.

Setelah model dan kelas sudah berhasil dibuat, langkah berikutnya adalah melakukan evaluasi model dengan melakukan visualisasi nilai metrik RMSE dan melihat daftar rekomendasi postingan berdasarkan postingan yang sering dilihat pengguna. Untuk menguji hasil rekomendasi postingan, model cukup diuji dengan satu sampel saja dengan bentuk keluaran yang berbeda.


Model yang telah dibuat dapat menghasilkan top-10 rekomendasi buku seperti yang ditunjukkan berikut ini.

tabel 3.rekomendasi user 134732.

|Showing recommendations for users: 134732 |
|---|
|Books with high ratings from user|
|Jennifer Egan - Look at Me|
|Jacquelyn Mitchard - The Deep End of the Ocean|
|Barry Unsworth - Morality Play|
|Matt Ridley - The Best American Science Writing 2002 (Best American Science Writing (Paperback))|
|Hari Kunzru - The Impressionist|

tabel 4.rekomendasi buku.
|Top 10 book recommendation|
|---|
|1. J. K. Rowling - Harry Potter and the Chamber of Secrets (Book 2)|
|2. Bill Watterson - The Authoritative Calvin and Hobbes (Calvin and Hobbes)|
|3. J.R.R. TOLKIEN - The Return of the King (The Lord of the Rings, Part 3)|
|4. J. K. Rowling - Harry Potter and the Goblet of Fire (Book 4)|
|5. J. R. R. Tolkien - The Two Towers (The Lord of the Rings, Part 2)|
|6. Bill Watterson - Calvin and Hobbes|
|7. DIANA GABALDON - Drums of Autumn|
|8. Jodi Picoult - My Sister's Keeper : A Novel (Picoult, Jodi)|
|9. Scott Adams - Dilbert: A Book of Postcards|
|10. J. K. Rowling - Harry Potter and the Chamber of Secrets Postcard Book|


## Evaluation
Pada proyek ini menggunakan metrik RMSE (Root Mean Square Error) untuk mengevaluasi kinerja model yang dihasilkan. RMSE adalah cara standar untuk mengukur kesalahan model dalam memprediksi data kuantitatif [[2](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)]. Root Mean Squared Error (RMSE) mengevaluasi model regresi linear dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. Perhitungan RMSE ditunjukkan pada rumus berikut ini.

![RMSE](https://i.postimg.cc/tgjfntZk/RMSE.png)

_Gambar 1.rumus rmse._

`RMSE` = nilai root mean square error

`y`  = nilai hasil observasi

`ŷ`  = nilai hasil prediksi

`i`  = urutan data

`n`  = jumlah data

Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.

Berikut ini adalah plot metrik RMSE setelah proses pelatihan model.

![Screenshot (147)](https://user-images.githubusercontent.com/69046629/192476257-32d169ea-fe09-4685-84ed-0bb62e36f039.png)

_Gambar 2.metriks._


Berdasarkan metriks tersebut menunjukkan bahwa model yang telah dibuat memiliki nilai Root Mean Squared Error (RMSE) sebesar 0.2550

## Kesimpulan

setelah melakukan semua proses dari mulai menyiapkan data sampai melakukan evaluasi, kita telah berhasil membuat sistem rekomendasi 10 buku kepada user dengan memanfaatkan data buku yang pernah diberi rating oleh user di masa lampau.


## Referensi

https://www.aindhae.com/2019/10/cara-menghitung-root-mean-squared-error.html

https://www.dicoding.com/academies/319/tutorials/19667?from=19662

https://developers.google.com/machine-learning/recommendation/collaborative/basics

