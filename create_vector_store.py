import os
import pandas as pd
# LangChain v0.1.16 için doğru import yolunu kullanıyoruz
from langchain_core.documents import Document 
# ChromaDB importu
from langchain_community.vectorstores import Chroma

# VertexAIEmbeddings importu kaldırıldı.

# Proje Kimliğiniz (Sadece LLM için kalacak)
PROJECT_ID = "genai-final-project-475415"

def load_and_prepare_data(filepath, sample_size=100):
    """CSV dosyasını okur ve LangChain dokümanlarına dönüştürür."""
    try:
        df = pd.read_csv(filepath).head(sample_size)
        documents = []
        for index, row in df.iterrows():
            content = (
                f"Recipe Title: {row['name']}\n\n"
                f"Ingredients: {row['ingredients']}\n\n"
                f"Instructions: {row['steps']}"
            )
            metadata = {'source': f"recipes.csv_row_{index}", 'recipe_name': row['name']}
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        return documents
    except Exception as e:
        print(f"Veri hazırlanırken hata oluştu: {e}")
        return None

def create_new_vector_store(documents, project_id):
    """Sıfırdan yeni bir vektör veritabanı oluşturur ve kaydeder."""
    if not documents:
        print("Doküman bulunamadı.")
        return
    try:
        print("Veritabanı oluşturuluyor... Bu işlem birkaç dakika sürebilir.")
        
        # !!! KRİTİK DEĞİŞİKLİK: Embeddings API çağrısı kaldırıldı !!!
        # ChromaDB kendi varsayılan (lokal, kurulum hatası vermeyen) embeddings modelini kullanacak.
        vector_store = Chroma.from_documents(
            documents, 
            persist_directory="chroma_db_recipes" # Yeni ve doğru klasör adı
        )
        vector_store.persist()
        
        print("\nBaşarılı! 'chroma_db_recipes' adında yeni ve temiz bir veritabanı oluşturuldu.")
    except Exception as e:
        # Hata mesajını daha anlaşılır yapıyoruz.
        print(f"\nVeritabanı oluşturulurken KRİTİK HATA oluştu: {e}")

if __name__ == "__main__":
    recipe_docs = load_and_prepare_data('recipes.csv')
    if recipe_docs:
        create_new_vector_store(recipe_docs, PROJECT_ID)