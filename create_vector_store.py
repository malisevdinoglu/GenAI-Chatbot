import json
import os
import pandas as pd
from langchain_core.documents import Document 
from langchain_community.vectorstores import Chroma
from google.oauth2 import service_account
from langchain_google_vertexai import VertexAIEmbeddings

# Proje Kimliğiniz (Sadece LLM için kalacak)
PROJECT_ID = "genai-final-project-475415"

def _resolve_vertex_credentials():
    """Env değişkenlerinden Vertex AI kimlik bilgilerini toparlar."""
    location = os.environ.get("GOOGLE_LOCATION", "us-central1")
    credentials = None

    service_account_payload = (
        os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        or os.environ.get("GOOGLE_SERVICE_ACCOUNT")
    )
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    try:
        if service_account_payload:
            if isinstance(service_account_payload, str):
                service_account_info = json.loads(service_account_payload)
            else:
                service_account_info = dict(service_account_payload)
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
        elif credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
    except Exception as exc:
        print(f"Servis hesabı kimlik bilgileri yüklenemedi: {exc}")

    return location, credentials

def _build_embeddings(project_id):
    """Chroma için Vertex AI embedding nesnesi hazırlar."""
    location, credentials = _resolve_vertex_credentials()
    return VertexAIEmbeddings(
        project=project_id,
        location=location,
        model_name="text-embedding-005",
        credentials=credentials,
    )

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
        
        embeddings = _build_embeddings(project_id)
        vector_store = Chroma.from_documents(
            documents, 
            embedding=embeddings,
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
