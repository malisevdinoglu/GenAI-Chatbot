import json
import os
import pandas as pd
# LangChain v0.1.16 için doğru import yolları
from langchain_core.documents import Document 
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# Google Cloud Kimlik Bilgileri için gerekli
from google.oauth2 import service_account

# Vertex AI bileşenleri
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI 

# Streamlit helper'ları ekleniyor (app.py ile tutarlı olmalı)
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    st = None

    def get_script_run_ctx():
        return None

PROJECT_ID = "genai-final-project-475415" 
DEFAULT_EMBEDDING_MODEL = os.environ.get("VERTEX_EMBEDDING_MODEL", "text-embedding-005")
DEFAULT_LLM_MODEL = os.environ.get("VERTEX_LLM_MODEL", "gemini-1.5-flash")

def _resolve_vertex_credentials():
    """Env'den konum ve kimlik bilgilerini çıkarır."""
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
        # Aksi halde ADC kullanılacak; credentials None kalır.
    except Exception as exc:
        print(f"Servis hesabı kimlik bilgileri yüklenemedi: {exc}")

    return location, credentials

def _build_embeddings(project_id):
    """Vertex AI metin embedding modelini hazırlar."""
    location, credentials = _resolve_vertex_credentials()
    return VertexAIEmbeddings(
        project=project_id,
        location=location,
        model_name=DEFAULT_EMBEDDING_MODEL,
        credentials=credentials,
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

def load_vector_store(project_id, index_path="chroma_db_recipes"): 
    """Kaydedilmiş Chroma vektör deposunu yükler (Lokal Embeddings kullanır)."""
    if not os.path.exists(index_path):
        print(f"Hata: '{index_path}' klasörü bulunamadı.") 
        return None
    try:
        embeddings = _build_embeddings(project_id)
        vector_store = Chroma(
            persist_directory=index_path,
            embedding_function=embeddings,
        ) 
        
        print("Vektör deposu başarıyla yüklendi.")
        return vector_store
    except Exception as e:
        print(f"Vektör deposu yüklenirken bir hata oluştu: {e}")
        _emit_streamlit_exception("Vektör deposu yüklenirken bir hata oluştu.", e)
        return None

def create_conversational_chain(project_id, vector_store):
    """Sohbet zincirini oluşturur (LLM için Vertex AI kullanır)."""
    try:
        location, credentials = _resolve_vertex_credentials()
        # LLM (Gemini) için Vertex AI kullanılmaya devam ediyor.
        llm = VertexAI(
            project=project_id,
            model_name=DEFAULT_LLM_MODEL,
            temperature=0.7,
            location=location,
            credentials=credentials,
        )
        
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        print("Sohbet zinciri başarıyla oluşturuldu.")
        return chain
    except Exception as e:
        print(f"Sohbet zinciri oluşturulurken bir hata oluştu: {e}")
        _emit_streamlit_exception("Sohbet zinciri oluşturulurken bir hata oluştu.", e)
        return None

# --- Ana Chatbot Çalışma Döngüsü ---
def main():
    if not PROJECT_ID or PROJECT_ID == "SENIN-PROJE-IDN":
        print("Lütfen kodun en üstüne geçerli bir Proje ID'si girin.")
        return

    vector_store = load_vector_store(project_id=PROJECT_ID)
    if not vector_store:
        return

    chain = create_conversational_chain(project_id=PROJECT_ID, vector_store=vector_store)
    if not chain:
        return

    print("\n--- Sağlıklı Tarif Asistanı ---")
    print("Merhaba! Ben sizin tarif asistanınızım. Nasıl bir tarif arıyorsunuz?")
    print("Çıkmak için 'çıkış' yazabilirsiniz.")

    while True:
        user_input = input("\nSiz: ")
        if user_input.lower() in ["çıkış", "quit", "exit"]:
            print("Görüşmek üzere!")
            break
        
        try:
            response = chain.invoke({'question': user_input})
            print(f"\nAsistan: {response['answer']}")
        except Exception as e:
            print(f"\nBir hata oluştu: {e}")
            break

if __name__ == "__main__":
    main()
