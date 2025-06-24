# 🦷 Cavity Analysis Web Platformu

Bu proje, diş hekimliği öğrencilerinin oluşturduğu 3D STL dosyalarının analizini yapan, sonuçları değerlendiren ve web tabanlı bir arayüzde gösteren tam entegre bir sistemdir. Projede Node.js, MySQL ve Python birlikte kullanılmaktadır.

## Özellikler

- Öğrenci kayıt sistemi
- STL dosyası yükleme
[image]
- Python destekli otomatik oyuk (cavity) analizi
- Web arayüzü üzerinden sonuçların görüntülenmesi
[image]
---

##  Kurulum Adımları

Aşağıdaki adımlar, projeyi yerel makinenizde çalıştırmak için gereken tüm işlemleri açıklar.

### 1. Python Gerekliliklerini Yükleme

Proje, bir Python scripti (`main.py`) aracılığıyla analiz gerçekleştirmektedir. Bu scriptin bağımlılıklarını yüklemek için:

```bash
pip install -r requirements.txt
```
## 2. Node.js Bağımlılıklarını Yükleme

Proje içerisinde Express, Multer, EJS ve MySQL gibi Node.js modülleri kullanılmaktadır. Aşağıdaki komutu çalıştırarak tüm bağımlılıkları yükleyebilirsiniz:

```bash
npm install
```


---


## 3. Veritabanı Kurulumu

Projede bir MySQL veritabanı kullanılmaktadır. Veritabanı ve gerekli tabloları oluşturmak için `init.sql` dosyasını çalıştırmanız yeterlidir.

### Adımlar:

1. MySQL servisinizin çalıştığından emin olun.
2. Terminalden aşağıdaki komutu çalıştırarak script'i yükleyin:

```bash
mysql -u root -p < init.sql
```
Not: init.sql dosyasında veritabanı adı `YOURDB` olarak tanımlanmıştır. Gerekirse `main.js` ve `main.py` kullanıcı/parola bilgilerini güncelleyiniz.

```bash
node main.js
```

