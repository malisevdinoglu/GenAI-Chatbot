import os

import pandas as pd
# LangChain v0.1.16 için doğru import yolunu kullanıyoruz
from langchain_core.documents import Document 
# ChromaDB importu
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

# Proje Kimliğiniz (Sadece LLM için kalacak)
PROJECT_ID = "genai-final-project-475415"

def build_embeddings(project_id, location=None, model_name="text-embedding-004"):
    """Vertex AI metin embedding modelini hazırlar."""
    location = location or os.environ.get("GOOGLE_LOCATION", "us-central1")
    return VertexAIEmbeddings(project=project_id, location=location, model_name=model_name)

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

def create_new_vector_store(documents, project_id, persist_directory="chroma_db_recipes"):
    """Sıfırdan yeni bir vektör veritabanı oluşturur ve kaydeder."""
    if not documents:
        print("Doküman bulunamadı.")
        return
    try:
        print("Veritabanı oluşturuluyor... Bu işlem birkaç dakika sürebilir.")
        
        embeddings = build_embeddings(project_id)

        vector_store = Chroma.from_documents(
            documents, 
            embedding=embeddings,
            persist_directory=persist_directory # Yeni ve doğru klasör adı
        )
        vector_store.persist()
        
        print(f"\nBaşarılı! '{persist_directory}' adında yeni ve temiz bir veritabanı oluşturuldu.")
    except Exception as e:
        # Hata mesajını daha anlaşılır yapıyoruz.
        print(f"\nVeritabanı oluşturulurken KRİTİK HATA oluştu: {e}")

if __name__ == "__main__":
    recipe_docs = load_and_prepare_data('recipes.csv')
    if recipe_docs:
        create_new_vector_store(recipe_docs, PROJECT_ID)
