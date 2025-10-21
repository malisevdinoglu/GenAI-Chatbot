import os
import time # Zamanlama için time modülünü ekliyoruz
import shutil # Klasör silme işlemi için

import pandas as pd
from google.oauth2 import service_account

try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    st = None

    def get_script_run_ctx():
        return None

# LangChain v0.1.16 için doğru import yolunu kullanıyoruz
from langchain_core.documents import Document
# ChromaDB importu
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

# Proje Kimliğiniz (Sadece LLM için kalacak)
PROJECT_ID = "genai-final-project-475415" # Kendi Proje ID'nizle değiştirin
# Konumu burada da belirtmek iyi bir pratik olabilir, app.py ile tutarlı olmalı
LOCATION = "us-central1" # Veya "europe-west1" gibi app.py'deki değer

def build_embeddings(project_id, location=None, model_name="text-embedding-004"): # Model adını güncelledik
    """Vertex AI metin embedding modelini hazırlar."""
    location = location or os.environ.get("GOOGLE_LOCATION", LOCATION) # Global LOCATION kullan
    # Kimlik bilgileri ADC veya ortam değişkeni üzerinden alınacak
    return VertexAIEmbeddings(
        project=project_id,
        location=location,
        model_name=model_name,
    )

def _emit_streamlit_exception(message, exception):
    """Streamlit oturumu varsa hatayı ekranda gösterir."""
    if st is None:
        return
    try:
        if get_script_run_ctx() is None:
            return
        st.error(message)
        st.exception(exception)
    except Exception:
        pass

def load_and_prepare_data(filepath, sample_size=1000): # sample_size'ı buradan ayarlayabilirsiniz
    """CSV dosyasını okur ve LangChain dokümanlarına dönüştürür."""
    try:
        df = pd.read_csv(filepath).head(sample_size)
        documents = []
        for index, row in df.iterrows():
            # Eksik veri kontrolü (name, ingredients, steps)
            if pd.isna(row.get('name')) or pd.isna(row.get('ingredients')) or pd.isna(row.get('steps')):
                print(f"\nUyarı: {index}. satırda eksik veri bulundu, atlanıyor.")
                continue # Eksik verili satırı atla
                
            content = (
                f"Recipe Title: {row['name']}\n\n"
                f"Ingredients: {row['ingredients']}\n\n"
                f"Instructions: {row['steps']}"
            )
            metadata = {'source': f"recipes.csv_row_{index}", 'recipe_name': row['name']}
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            # İlerlemeyi gösterme (her satır için değil, belirli aralıklarla daha verimli olabilir)
            if (index + 1) % 50 == 0: 
                 print(f"\rVeri hazırlanıyor: {index + 1}/{sample_size}", end="")
        print(f"\rVeri hazırlama tamamlandı: {len(documents)} doküman hazırlandı.") # Son durumu yazdır
        return documents
    except FileNotFoundError:
        print(f"\nHata: '{filepath}' dosyası bulunamadı.")
        return None
    except Exception as e:
        print(f"\nVeri hazırlanırken hata oluştu: {e}")
        return None

def create_new_vector_store(documents, project_id, persist_directory="chroma_db_recipes"):
    """Sıfırdan yeni bir vektör veritabanı oluşturur ve kaydeder (Gecikmeli)."""
    if not documents:
        print("\nDoküman bulunamadı.")
        return
        
    # Başlamadan önce eski klasörü sil (varsa)
    if os.path.exists(persist_directory):
        print(f"\nEski '{persist_directory}' klasörü siliniyor...")
        shutil.rmtree(persist_directory)
        
    try:
        print("\nVeritabanı oluşturuluyor... Bu işlem birkaç dakika sürebilir.")
        
        embeddings = build_embeddings(project_id)

        # !!! DEĞİŞİKLİK BURADA: Dokümanları gruplar halinde işleme !!!
        vector_store = None
        batch_size = 50 # Her seferinde kaç doküman işlenecek (API limitine göre ayarlanabilir)
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:min(i+batch_size, total_docs)] # Son grubu doğru almak için min() kullanıldı
            
            if not batch: # Eğer batch boşsa döngüye devam et
                continue

            if vector_store is None:
                # İlk grup için veritabanını oluştur
                vector_store = Chroma.from_documents(
                    batch, 
                    embedding=embeddings,
                    persist_directory=persist_directory
                )
            else:
                # Sonraki grupları mevcut veritabanına ekle
                vector_store.add_documents(batch, embedding=embeddings) # Chroma'da embedding tekrar belirtmeye gerek yok
            
            processed_count = min(i + batch_size, total_docs)
            print(f"\rİşlenen grup: {i//batch_size + 1} / {(total_docs + batch_size - 1)//batch_size} | Toplam doküman: {processed_count}/{total_docs}", end="")
            
            # !!! DEĞİŞİKLİK BURADA: Her gruptan sonra kısa bir süre bekle !!!
            # Kota hatası almamak için 1 saniye bekle (Gerekirse artırılabilir)
            time.sleep(1) 

        # İşlem bittikten sonra veritabanını diske kaydet (Chroma bunu zaten yapıyor olabilir ama garanti olsun)
        if vector_store:
            vector_store.persist()
            print(f"\n\nBaşarılı! '{persist_directory}' adında yeni ve temiz bir veritabanı oluşturuldu.")
        else:
            print("\nHiçbir doküman işlenemediği için veritabanı oluşturulamadı.")

    except Exception as e:
        # Hata mesajını daha anlaşılır yapıyoruz.
        print(f"\nVeritabanı oluşturulurken KRİTİK HATA oluştu: {e}")
        _emit_streamlit_exception("Veritabanı oluşturulurken bir hata oluştu. Ayrıntılar için logları kontrol edin.", e)

if __name__ == "__main__":
    # sample_size'ı ihtiyacınıza göre ayarlayabilirsiniz
    recipe_docs = load_and_prepare_data('recipes.csv', sample_size=300) 
    if recipe_docs:
        create_new_vector_store(recipe_docs, PROJECT_ID)