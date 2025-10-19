import os
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings

# Proje Kimliğiniz (Project ID)
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

# create_vector_store.py dosyasındaki create_new_vector_store fonksiyonu

def create_new_vector_store(documents, project_id):
    """Sıfırdan yeni bir vektör veritabanı oluşturur ve kaydeder."""
    if not documents:
        print("Doküman bulunamadı.")
        return
    try:
        print("Veritabanı oluşturuluyor... Bu işlem birkaç dakika sürebilir.")
        
        # Embedding için bölgeyi sabitleyelim ve kodun hata vermemesi için basit tutalım.
        # Bu, Service Account (Secrets) yetkilendirmesi ile çalışmalıdır.
        embeddings = VertexAIEmbeddings(project=project_id, model_name="text-embedding-004", location="us-central1") 
        
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local("faiss_index_recipes")
        
        print("\nBaşarılı! 'faiss_index_recipes' adında yeni ve temiz bir veritabanı oluşturuldu.")
    except Exception as e:
        # Hata mesajını daha anlaşılır yapıyoruz.
        print(f"\nVeritabanı oluşturulurken KRİTİK HATA oluştu: {e}")

if __name__ == "__main__":
    recipe_docs = load_and_prepare_data('recipes.csv')
    if recipe_docs:
        create_new_vector_store(recipe_docs, PROJECT_ID)