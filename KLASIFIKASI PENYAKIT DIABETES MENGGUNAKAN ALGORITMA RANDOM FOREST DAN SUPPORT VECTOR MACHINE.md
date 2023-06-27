# KLASIFIKASI PENYAKIT DIABETES MENGGUNAKAN ALGORITMA RANDOM FOREST DAN SUPPORT VECTOR MACHINE


### DOMAIN PROJEK
Penyakit diabetes merupakan salah satu penyakit yang berbahaya di dunia. Menurut World Health Organization pada tahun 2020, penyakit diabetes masuk dalam sepuluh besar penyebab utama kematian secara global.Menurut International Diabetes Federation (IDF), prevalensi penyakit diabetes di dunia pada tahun 2021 sebanyak 536,6 juta, dan akan meningkat sebanyak 11,3% menjadi 642,7 juta pada tahun 2030, dan 12,2% menjadi 783,2 juta pada tahun 2045. Selain itu, Indonesia diprediksi akan menempati ranking kelima dimana sebanyak 28,6 juta penduduk Indonesia akan terkena penyakit ini pada tahun 2045.
Walaupun tidak menular, penyakit diabetes dapat menyerang siapa saja. Oleh karena itu, masyarakat perlu untuk mewaspadai adanya kemungkinan terkena penyakit ini. Oleh karena itu, dengan data yang ada, kemungkinan terjadinya penyakit diabetes dapat diketahui dengan cepat. Data yang ada dapat dimanfaatkan secara maksimal, salah satunya yaitu klasifikasi dalam machine learning. Klasifikasi dapat didefinisikan sebagai teknik yang mempelajari tentang sekumpulan data sehingga mendapatkan aturan yang dapat mengenali data baru yang belum dipelajari sebelumnya. Penelitian dengan data penyakit diabetes juga telah dilakukan oleh Nugraha & Sabaruddin (2021). Dataset yang digunakan yaitu Pima Indians Diabetes, dengan total variabel sebanyak 9, dimana 1 variabelnya merupakan output/label. Dari penelitian tersebut didapatkan hasil bahwa dengan metode Random Forest akurasinya sebesar 75,82% (oversampling), 71,24% (undersampling), dan 73,86 (original). Penelitian lainnya juga telah dilakukan oleh Lyngdoh et al., (2020) menggunakan data diabetes dengan tujuh variabel dan beberapa algoritma seperti K-Nearest Neigbour, Naïve Bayes, dan Decision Tree. Dari penelitian itu didapatkan kesimpulan bahwa K-Nearest Neighbor menghasilkan akurasi yang paling baik diantara algoritma yang lainnya, yaitu sebesar 76%. Akan tetapi, keakuratan tersebut belum terlalu baik.

## BUSINESS UNDERSTANDING
Dalam prediksi diagnosa, machine learning merupakan metode yang menjanjikan untuk pengembangan deteksi penyakit. Metode tersebut disusun sehingga mampu mengeksplorasi sebuah data, menemukan sebuah pola, dan membantu menemukan sebuah pengetahuan yang baru. Data yang ada dapat dimanfaatkan secara maksimal, salah satunya yaitu klasifikasi. Klasifikasi merupakan metode pengelompokan data yang memiliki kelas atau target. Menurut Sutoyo dan Fadlurrahman (2020), klasifikasi merupakan salah satu salah satu fungsi dari data mining untuk mengelompokkan suatu item data ke dalam kategori atau kelas yang telah didefinisikan. Klasifikasi ini dilakukan dengan tujuan untuk memperkirakan kelas dari suatu objek yang labelnya belum diketahui sebelumnya.Cara kerjanya yaitu algoritma klasifikasi mempelajari data training yang berisikan data dan label yang telah diklasifikasikan terlebih dahulu. Algoritma berisi pembelajaran akan belajar dengan data yang ada dan menemukan sebuah hubungan antara data input dan output/label. Selanjutnya algoritma melakukan klasifikasi pada data testing yang hanya berisi data input tanpa label berdasarkan pengalaman mempelajari data training. Algoritma yang populer dalam klasifikasi adalah algoritma Random Forest. Random Forest merupakan salah satu metode klasifikasi yang terdiri dari kumpulan pohon keputusan. Random Forest memiliki keunggulan  dapat memberikan akurasi yang tinggi. Selain Random Forest, terdapat algoritma Support Vector Machine yang juga bisa digunakan dalam klasifikasi. Support Vector Machine memiliki kelebihan dimana algoritma ini mempunyai kemampuan generalisasi yang tinggi dan dapat menghasilkan model klasifikasi yang baik. SVM memiliki kelebihan seperti dapat bekerja mengklasifikasikan data yang linier ataupun nonlinier. Algoritma SVM dapat melakukan generalisasi dengan klasifikasi data lain yang tidak termasuk ke dalam data yang digunakan.

### Problem Statements
1.	Bagaimana perbandingan klasifikasi penyakit diabetes menggunakan algoritma Random Forest dan Support Vector Machine (SVM)?
2.	Bagaimana performa Random Forest dan Support Vector Machine (SVM) dalam klasifikasi penyakit diabetes, serta manakah yang lebih baik?
3. Variabel apa yang paling berpengaruh dalam klasifikasi penyakit diabetes ini?

### Goals
1.	Membandingkan klasifikasi penyakit diabetes menggunakan algoritma Random Forest dan Support Vector Machine (SVM).
2.	Menentukan performansi hasil klasifikasi penyakit diabetes menggunakan algoritma Random Forest dan Support Vector Machine (SVM).
3.	Mengetahui  variabel yang paling berpengaruh dalam klasifikasi penyakit diabetes ini.

## Data Understanding
Data yang digunakan dalam projek ini merupakan data sekunder dari Rumah Sakit Shyllet, Bangladesh pada tahun 2020 yang diperoleh melalui situs Kaggle. Data ini berjumlah sebanyak 520, dan  memiliki 17 variabel, di mana 1 variabelnya merupakan label. Dari data ini terdapat 200 data tidak menderita penyakit diabetes dan 320 data menderita penyakit diabetes. Data ini diambil melalui situs Kaggle (https://www.kaggle.com/datasets/alakaaay/diabetes-uci-dataset?resource=download)

Variabel yang ada pada data yaitu:
1) Age: 20-65
2) Sex: Male/Female
3) Polyuria: Yes/No
4) Polydipsia: Yes/No
5) sudden weight loss: Yes/No
6) weakness: Yes/No
7) Polyphagia: Yes/No
8) Genital thrush: Yes/No
9) visual blurring: Yes/No
10) Itching: Yes/No
11) Irritability: Yes/No
12) delayed healing: Yes/No
13) partial paresis: Yes/No
14) muscle stiffness: Yes/No
15) Alopecia: Yes/No
16) Obesity: Yes/No
17) Class: Positive/Negative


## Data Preparation
CEK MISSING VALUE.
```sh
data.isna().sum()
```
Dari pengecekan tersebut tidak ada missing value.

Selanjutnya yaitu melakukan label encoding. Data yang bernilai “tidak”, “Laki-Laki”, dan “Negatif”  akan diubah dengan label “-1”, dan data yang bernilai “Ya”, “Perempuan”, dan “Positif” akan diubah menjadi “1”. Label encoding dilakukan karena algoritma support vector machine merupakan algoritmma yang basisnya menggunakan jarak, sehingga data kategorik harus diubah menjadi numerik, selain itu pada python dalam klasifikasi tidak dapat mengolah data kategorik.

Langkah selanjutnya yaitu melihat apakah terdapat outliers pada data. Dari semua variabel yang ada, terdapat outliers pada variabel usia (karena data lainnya hanya berisi 2 tipe, ya/tidak, atau laki laki/perempuan). Outliers dapat dilihat melalui Gambar dibawah ini. 
```sh
## CEK OUTLIERS
sns.boxplot(data=data, x=data['Age'])
plt.title("Boxplot Age")
```
![Gambar outliers](https://drive.google.com/file/d/1lAkf-dfqtRycDvhqMEKWmdq5PUFqviWa/view)


Outliers pada variabel Age ini tidak dibuang. Hal itu dikarenakan outliers tersebut merupakan fenomena dari subjek penelitian, sehingga tidak dilakukan pembuangan data.

Langkah selanjutnya yaitu dilakukan pengecekan keseimbangan data.
![Gambar Keseimbangan Data](https://drive.google.com/file/d/1lAkf-dfqtRycDvhqMEKWmdq5PUFqviWa/view?usp=sharing)

Gambar tersebut memperlihatkan bahwa data tidak seimbang. Oleh karena itu, perlu dilakukan penanganan keseimbangan data. Data dapat diseimbangkan menggunakan Synthetic Minority Oversampling Technique (SMOTE). Data tersebut diseimbangkan setelah data di split menjadi data training dan data testing. 

## MODELLING
SPLIT DATA
```sh
#drop 
X = data.drop(['class'], axis = 1)
y = data['class']

#SPLIT DATA 80 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42, stratify = y)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '-1': {} \n".format(sum(y_train==-1)))
```
Pada kondisi split dengan rasio 80%:20%, jumlah data training dengan label -1 sebanyak 160 dan label 1 sebanyak 256. Karena data tidak seimbang, maka data diseimbangkan dengan SMOTE.
```sh
#SMOTE
smote=SMOTE(sampling_strategy = 0.9)
X_oversampled,y_oversampled=smote.fit_resample(X_train,y_train)

print("after OverSampling, counts of label '1': {}".format(sum(y_oversampled==1)))
print("after OverSampling, counts of label '-1': {} \n".format(sum(y_oversampled==-1)))

#PLOT SETELAH SMOTE
labels = y_oversampled.value_counts(sort = True).index
sizes =  y_oversampled.value_counts(sort = True)
colors = ["lightblue","red"]
explode = (0.05,0) 
plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,)
plt.title('Data Training 80%:20% Setelah Dilakukan Balancing Data')
plt.show()
```
![GAMBAR SETELAH DILAKUKAN SMOTE](https://drive.google.com/file/d/1OxQSiWKeHqbIrPCqWFDe6yIXzoDDZsAJ/view?usp=sharing)
terlihat bahwa data sudah seimbang. Data dengan label -1 setelah diseimbangkan menjadi sebanyak 230 data

### Klasifikasi Menggunakan Algoritma Random Forest
Algoritma Random Forest merupakan salah satu ensemble learning. Ensemble learning adalah metode dimana model akan dilatih untuk memecahkan masalah yang sama dan digabungkan untuk mendapatkan suatu hasil yang lebih baik. Algoritma Random Forest ini ialah algoritma yang dikembangkan dari algoritma Decision Tree. Decision tree adalah algoritma yang berbentuk sebuah pohon untuk mengambil kesimpulan. Decision Tree ini dapat digunakan untuk mengklasifikasikan sebuah data dengan variabel input dan output dalam bentuk pohon. Terdapat beberapa istilah yang digunakan dalam Decision Tree, yaitu root node, internal node, dan leaf. Root merupakan node yang terletak pada bagian paling atas di pohon. Internal node merupakan node percabangan yang masih memiliki cabang di bawahnya, sedangkan leaf merupakan node akhir yang tidak memiliki percabangan lagi. Decision Tree akan memasukkan sebuah input melalui root, dan memiliki kesimpulan melalui leaf node untuk menentukan data input termasuk dalam kelas yang mana. Algoritma ini dikembangkan menjadi sebuah algoritma baru yang dinamakan sebagai Random Forest. Sesuai dengan namanya, algoritma ini akan menciptakan sebuah hutan dengan sejumlah pohon. Cara kerja klasifikasi menggunakan algoritma ini yaitu Random Forest akan melakukan bootstrap pada data training untuk membentuk setiap pohon. Selanjutnya, pohon tersebut akan digabungkan dengan pohon yang lain, dimana satu pohon akan menghasilkan satu keputusan. Oleh sebab itu, algorima Random Forest ini dapat dikatakan sebagai kumpulan Decision Tree. Untuk mendapatkan hasil akhir, maka dilakukan majority voting dimana vote terbanyak akan menjadi pemenangnya. Menurut Pamuji dan Ramadhan (2021), algoritma Random Forest ini memiliki kelebihan yaitu dapat menghasilkan eror yang relatif rendah, performa yang baik dalam klasifikasi, dan cocok untuk data yang berjumlah besar. Parameter yang dapat digunakan dalam algoritma ini yaitu n estimator (jumlah pohon), max feature (jumlah variabel yang perlu dipertimbangkan saat mencari pemisah terbaik), max depth (kedalaman pohon), dan lain lain.

Hasil kinerja dari Random Forest dapat dilihat melalui gambar ini.
![Confusion matrix Random Forest](https://drive.google.com/file/d/17ZkShBRuRl-1SHE9V0Fmqqe9tg9TQzCB/view?usp=sharing)

Feature Importance Random Forest
feature importance Random Forest dapat dilihat melalui gambar dibawah ini
![Feature Importance Random Forest](https://drive.google.com/file/d/1VVN6SRJgcMVkwQgZHyLWzkV3KaVHNQ2R/view?usp=sharing)


### KLASIFIKASI MENGGUNAKAN ALGORITMA SUPPORT VECTOR MACHINE
Support vector machine (SVM) merupakan salah satu algoritma yang dapat digunakan dalam klasifikasi. SVM menggunakan hyperplane dengan pemisah dari pengelompokan kelas yang dibentuk dari suatu dimensi vektor berukuran n (Iman dan Wijayanto, 2021). Hyperplane terbaik didapatkan dengan mengukur margin hyperplane atau jarak antara vektor yang paling dekat dengan hyperplane. 

Pada SVM dilakukan normalisasi data. Hal ini dilakukan karena svm bekerja menggunakan jarak, dimana data akan dipisahkan menggunakan hyperplane. Terdapat variabel usia yang memiliki nilai besar dibandingkan dengan variabel yang lainnya. Oleh karena itu dilakukan normalisasi data. 
Hasil klasifikasi menggunakan SVM dapat dilihat melalui confusion matrix dibawah ini.
![Confusion matrix svm](https://drive.google.com/file/d/1SK1ndbypI3rj6NAIukizY6kqExCw0Q7-/view?usp=sharing)

# EVALUSASI
Berdasarkan hasil klasifikasi, akan dilakukan perbandingan dari kedua model tersebut. Melalui confusion matrix, dapat dihitung evaluation matrix untuk menilai kinerja dari sebuah algoritma klasifikasi, yaitu akurasi, presisi, recall, dan f1 score.
Evaluation matrix dari Random forest dapat dilihat melalui gambar dibawah ini. 
![Random Forest](https://drive.google.com/file/d/1WmoQiBIXeUoE5u0cyfnpFLmBYYqXOOL7/view?usp=sharing)
Dari gambar tersebut dapat diketahui bahwa akurasi, presisi, recall, dan f1 score dari random forest sebesar 0,99. Sedangkan untuk SVM evaluation matrix dapat dilihat melalui gambar dibawah ini.
![SVM](https://drive.google.com/file/d/1IOD1ZkbuWw1vL_ZN1AoMczF5ihaujoV5/view?usp=sharing)
Dari gambar tersebut dapat diketahui bahwa akurasi, presisi, recall, dan f1 score dari svm sebesar 0,98.

Dari kedua algoritma yang sudah digunakan, dapat dibandingkan bahwa algoritma random forest memiliki performa yang lebih bagus dibandingkan dengan SVM meskipun perbedaannya tidak terlalu jauh.
Selain itu, dari algoritma Random Forest dapat diketahui variabel yang berpengaruh terhadap penyakit diabetes ini. Tiga variabel yang paling berpengaruh terhadap penyakit diabetes ini secara berturut turut yaitu Polydipsia, Polyuria, dan Gender.


# REFERENSI
IDF. (2021). IDF Diabetes Atlas 10th Edition. www.diabetesatlas.org diakses pada tanggal 8 Agustus 2022.
Iman, Q. & Wijayanto, A. W. (2021). Klasifikasi Rumah Tangga Penerima Beras Miskin (Raskin)/Beras Sejahtera (Rastra) di Provinsi Jawa Barat Tahun 2017 dengan Metode Random Forest dan Support Vector Machine. JUSTIN (Jurnal Sistem dan Teknologi Informasi), 9(2), 178-184.
Lyngdoh, A. C., Choudhury, N. A., & Moulik, S. (2021, March). Diabetes Disease Prediction Using Machine Learning Algorithms. In 2020 IEEE-EMBS Conference on Biomedical Engineering and Sciences (IECBES) (pp. 517-521). IEEE.
Nugraha, W., & Sabaruddin, R. (2021). Teknik Resampling untuk Mengatasi Ketidakseimbangan Kelas pada Klasifikasi Penyakit Diabetes Menggunakan C4. 5, Random Forest, dan SVM. Techno. Com, 20(3), 352-361.
Pamuji, F. Y. & Ramadhan, V. P. (2021). Komparasi Algoritma Random Forest dan Decision Tree untuk Memprediksi Keberhasilan Immunotheraphy. Jurnal Teknologi dan Manajemen Informatika, 7(1), 46-50.
Sutoyo, E., & Fadlurrahman, M. A. (2020). Penerapan SMOTE untuk Mengatasi Imbalance Class dalam Klasifikasi Television Advertisement Performance Rating Menggunakan Artificial Neural Network. JEPIN (Jurnal Edukasi dan Penelitian Informatika), 6(3), 379-385.
WHO. (2020). The Top 10 Causes of Death. Diakses dari https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death 
WHO. (2021). Diabetes. Diakses dari  https://www.who.int/health-topics/diabetes#tab=tab_1 
WHO. (2022). Diabetes. Diakses dari https://www.who.int/news-room/fact-sheets/detail/diabetes
