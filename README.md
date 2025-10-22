# Sağlıklı Tarif Asistanı Chatbot

## Akbank GenAI Bootcamp Bitirme Projesi

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, RAG (Retrieval-Augmented Generation) mimarisi kullanılarak oluşturulmuş bir tarif asistanı chatbot'udur.

Uygulamanın canlı versiyonuna aşağıdaki linkten erişebilirsiniz:

https://genai-chatbot-jc59.onrender.com


# Proje Açıklaması

Bu projenin temel amacı, kullanıcılara geniş bir tarif veri seti içerisinden doğal dil sorguları ile arama yapma imkanı sunan bir yapay zeka asistanı geliştirmektir. Kullanıcılar, "içinde tavuk olan bir tarif" veya "yapması 30 dakikadan az süren bir tatlı" gibi karmaşık sorular sorarak, veri setindeki en uygun tariflere hızlıca ulaşabilirler. Bu çözüm, geleneksel anahtar kelime tabanlı arama sistemlerinin yetersiz kaldığı durumlarda, anlamsal arama yaparak daha doğru ve sezgisel sonuçlar üretir.

## Veri Seti

Bu projede, Kaggle üzerinde halka açık olarak yayınlanan **"Food.com Recipes and User Interactions"** veri seti kullanılmıştır ([Kaynak Linki - Opsiyonel: İsterseniz Kaggle linkini buraya ekleyebilirsiniz]). Orijinal veri seti, 200.000'den fazla yemek tarifini ve kullanıcı etkileşimlerini içermektedir. Her bir tarif; tarif adı (`name`), malzemeler (`ingredients`), hazırlama adımları (`steps`), toplam süre (`minutes`) ve besin değerleri (`nutrition`) gibi zengin bilgiler barındırmaktadır.

Projenin deploy edilebilir boyutlarda kalması ve vektör veritabanı oluşturma işleminin makul sürelerde tamamlanması amacıyla, bu orijinal veri setinden alınan **300 tariflik bir alt küme** kullanılmıştır (`sample_size=1000`). Bu alt küme, `create_vector_store.py` scripti tarafından işlenerek tariflerin anlamsal vektörlere dönüştürülmesini ve chatbot'un bilgi kaynağı olan ChromaDB veritabanının oluşturulmasını sağlamıştır.


## Çözüm Mimarisi (RAG)

Proje, güncel ve güçlü bir yapay zeka mimarisi olan ***Retrieval-Augmented Generation (RAG)*** üzerine kurulmuştur. Bu mimari, büyük dil modellerinin (LLM) yaratıcılığını, harici bir bilgi tabanının (vektör veritabanı) doğruluğu ile birleştirir.

**Çalışma Akışı:**
1.  **Veri Hazırlama (Indexing):** `recipes.csv` dosyasındaki tarifler okunur, her bir tarif anlamsal olarak zengin metinlere dönüştürülür ve Google'ın `text-embedding-005` modeli kullanılarak vektörlere çevrilir. Bu vektörler, hızlı arama için bir **ChromaDB** vektör veritabanına kaydedilir. Bu işlem `create_vector_store.py` scripti ile bir defaya mahsus yapılır.
2.  **Sorgu (Retrieval):** Kullanıcı web arayüzünden bir soru sorduğunda, bu soru da aynı embedding modeli ile bir vektöre dönüştürülür.
3.  **Benzerlik Arama:** Kullanıcının soru vektörü, FAISS veritabanındaki tarif vektörleri ile karşılaştırılır ve anlamsal olarak en alakalı tarifler (örneğin en benzer ilk 3 tarif) bulunur.
4.  **Cevap Üretme (Generation):** Bulunan bu alakalı tarifler, kullanıcı sorusu ile birlikte bir *prompt* (komut) olarak **Google Gemini (`gemini-2.5-flash`)** modeline gönderilir.
5.  **Sonuç:** Gemini modeli, kendisine sağlanan bu tarif bilgilerini kullanarak kullanıcıya bağlama uygun, doğal ve akıcı bir cevap üretir. Bu sayede model, bilmediği bir konuda "halüsinasyon" görmek yerine, doğrudan veri setimizdeki gerçek bilgilere dayalı cevaplar verir.


## Kullanılan Teknolojiler

* **Programlama Dili:** Python 3.11+
* **Web Arayüzü:** Streamlit
* **LLM & Embedding:** Google Gemini & Google Text Embedding Models (Vertex AI)
* **Orkestrasyon:** LangChain
* **Vektör Veritabanı:** ChromaDb
* **Veri İşleme:** Pandas
* **Deployment:** Render

## Kurulum ve Yerel Ortamda Çalıştırma

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

**1. Projeyi Klonlayın:**
```bash
git clone [https://github.com/malisevdinoglu/GenAI-Chatbot]
cd [GenAI-Chatbot]
```

**2. Sanal Ortam Oluşturun ve Aktif Edin:**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux için
# venv\Scripts\activate  # Windows için
```

**3. Gerekli Kütüphaneleri Yükleyin:**
```bash
pip install -r requirements.txt
```

**4. Google Cloud Kimlik Doğrulaması:**
Projenin Google Cloud servislerine erişebilmesi için kimlik doğrulaması yapmanız gerekmektedir.
```bash
# Google Cloud CLI'da oturum açın
gcloud auth login

# Uygulama varsayılan kimlik bilgilerini ayarlayın
gcloud auth application-default login
```
Bu komutlar sizi tarayıcıya yönlendirecek ve Google hesabınızla giriş yapmanızı isteyecektir.

**5. Vektör Veritabanını Oluşturun:**
Uygulamayı çalıştırmadan önce, tarif verilerinden vektör veritabanını oluşturmanız gerekmektedir.
```bash
python create_vector_store.py
```
Bu işlem birkaç dakika sürebilir ve tamamlandığında `chroma_db_recipes` adında bir klasör oluşturulacaktır.

**6. Uygulamayı Başlatın:**
Artık Streamlit uygulamasını başlatabilirsiniz.
```bash
streamlit run app.py
```
Uygulama, tarayıcınızda  açılacaktır.

## Ürün Kılavuzu (Nasıl Kullanılır?)

Uygulamanın web arayüzü oldukça basittir.

**1. Soru Sorma:**
Ana ekrandaki metin kutusuna tariflerle ilgili merak ettiğiniz bir soruyu yazın.
*Örnek Sorular:*
* "Bana içinde tavuk ve mantar olan bir tarif öner."
* "Vejetaryen bir makarna tarifi var mı?"
* "Yapması 15 dakikadan kısa süren bir tatlı arıyorum."

[Uygulama Ekran Görüntüsü](Chatbot-screen.png)

**2. Cevap Alma:**
Sorunuzu yazdıktan sonra "Gönder" butonuna tıklayın. Yapay zeka asistanı, veri setindeki en uygun tarifi bularak size bir cevap üretecektir.
ChatBot'un verdiği cevabın örnek bir ekran görüntüsü 
(Chatbot-screen.png)

## Elde Edilen Sonuçlar ve Gözlemler

Geliştirilen RAG tabanlı chatbot, `recipes.csv` veri setindeki bilgilerle sınırlı olmasına rağmen oldukça başarılı sonuçlar vermektedir.

* **Başarılar:** Model, malzemelere, pişirme sürelerine ve tarif türlerine göre sorulan spesifik sorulara doğru ve bağlama uygun cevaplar üretebilmektedir.
* **Limitasyonlar:** Veri setinde bulunmayan bir malzeme veya tarif türü sorulduğunda, model "Bu konuda bir bilgim yok" şeklinde cevap vermektedir. Bu, RAG mimarisinin halüsinasyonları önlemedeki başarısını göstermektedir.


## Proje Sahibi
Mehmet Ali Sevdinoğlu
* Linkedin Profil : [www.linkedin.com/in/mehmet-ali-sevdinoğlu-983179252]
* GitHub Profil : [https://github.com/malisevdinoglu] 

Projemin Canlı Linki: https://genai-chatbot-jc59.onrender.com

