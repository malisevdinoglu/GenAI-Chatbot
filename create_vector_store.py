import os
import pandas as pd
from langchain_core.documents import Document 
from langchain_community.vectorstores import Chroma
# VertexAIEmbeddings importu ve bağımlılıkları kaldırıldı.

# Proje Kimliğiniz (Sadece LLM için kalacak)
PROJECT_ID = "genai-final-project-475415"

def load_and_prepare_data(filepath, sample_size=100):
    """CSV dosyasını okur ve LangChain dokümanlarına dönüştürür."""
    try:
        df = pd.read_csv(filepath).head(sample_size)
        documents = []
        for index, row in df.iterrows():
            # Eksik veri kontrolü
            if pd.isna(row.get('name')) or pd.isna(row.get('ingredients')) or pd.isna(row.get('steps')):
                print(f"\nUyarı: {index}. satırda eksik veri bulundu, atlanıyor.")
                continue
                
            content = (
                f"Recipe Title: {row['name']}\n\n"
                f"Ingredients: {row['ingredients']}\n\n"
                f"Instructions: {row['steps']}"
            )
            metadata = {'source': f"recipes.csv_row_{index}", 'recipe_name': row['name']}
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        print(f"\rVeri hazırlama tamamlandı: {len(documents)} doküman hazırlandı.") 
        return documents
    except FileNotFoundError:
        print(f"\nHata: '{filepath}' dosyası bulunamadı.")
        return None
    except Exception as e:
        print(f"\nVeri hazırlanırken hata oluştu: {e}")
        return None

def create_new_vector_store(documents, project_id, persist_directory="chroma_db_recipes"):
    """Sıfırdan yeni bir vektör veritabanı oluşturur ve kaydeder."""
    if not documents:
        print("\nDoküman bulunamadı.")
        return
        
    try:
        print("\nVeritabanı oluşturuluyor... Bu işlem birkaç dakika sürebilir.")
        
        # !!! KRİTİK DEĞİŞİKLİK: Embeddings API çağrısı kaldırıldı !!!
        # ChromaDB kendi varsayılan (lokal) embeddings modelini kullanacak.
        vector_store = Chroma.from_documents(
            documents, 
            persist_directory=persist_directory 
        )
        vector_store.persist()
        
        print(f"\n\nBaşarılı! '{persist_directory}' adında yeni ve temiz bir veritabanı oluşturuldu.")
    except Exception as e:
        # Hata mesajını yakalayacak en basit yapı budur.
        print(f"\nVeritabanı oluşturulurken KRİTİK HATA oluştu: {e}")

if __name__ == "__main__":
    # sample_size'ı bulut ortamına göre küçük tutuyoruz
    recipe_docs = load_and_prepare_data('recipes.csv', sample_size=100) 
    if recipe_docs:
        create_new_vector_store(recipe_docs, PROJECT_ID)