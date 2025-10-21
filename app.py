import json
import os
from pathlib import Path

import streamlit as st
from google.oauth2 import service_account
import vertexai

# ChromaDB ve doğru embedding importları
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI

# Diğer LangChain importları
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document # load_and_prepare_data için gerekli

# create_vector_store.py'dan gerekli fonksiyonları import ediyoruz
# (Bunlar zaten ChromaDB kullanıyor olmalı)
from create_vector_store import load_and_prepare_data, create_new_vector_store 

# Proje Kimliğiniz ve Varsayılan Model Ayarları
PROJECT_ID = "genai-final-project-475415" # Kendi Proje ID'nizle değiştirin
DEFAULT_EMBEDDING_MODEL = os.environ.get("VERTEX_EMBEDDING_MODEL", "text-embedding-005")
DEFAULT_LLM_MODEL = os.environ.get("VERTEX_LLM_MODEL", "gemini-1.5-flash")

def _get_streamlit_secret(key):
    """Streamlit secrets'tan güvenli şekilde okur; yoksa None döner."""
    if not hasattr(st, "secrets"):
        return None
    try:
        return st.secrets[key]
    except Exception:
        return None

def configure_google_credentials():
    """Render veya Streamlit gibi ortamlar için kimlik bilgilerini hazırlar."""
    project_id = os.environ.get("GOOGLE_PROJECT_ID", PROJECT_ID) # Doğrudan PROJECT_ID kullan
    location = os.environ.get("GOOGLE_LOCATION", "us-central1") # Varsayılan konumu ayarlayalım
    service_account_payload = (
        os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        or os.environ.get("GOOGLE_SERVICE_ACCOUNT")
    )

    secret_project_id = _get_streamlit_secret("GOOGLE_PROJECT_ID")
    secret_location = _get_streamlit_secret("GOOGLE_LOCATION")
    secret_service_account = _get_streamlit_secret("GOOGLE_SERVICE_ACCOUNT")

    if secret_project_id:
        project_id = project_id or secret_project_id
    if secret_location:
        location = os.environ.get("GOOGLE_LOCATION", secret_location)
    if secret_service_account and not service_account_payload:
        secret_value = secret_service_account
        if isinstance(secret_value, str):
            service_account_payload = secret_value
        else:
            service_account_payload = json.dumps(dict(secret_value))

    # Lokal çalıştırma için gcloud ADC kullan (Payload yoksa)
    credentials = None
    if service_account_payload:
        try:
            if isinstance(service_account_payload, str):
                service_account_info = json.loads(service_account_payload.strip())
            else:
                service_account_info = dict(service_account_payload)
            
            if not service_account_info:
                 raise ValueError("Servis hesabı bilgisi boş.")
            if not isinstance(service_account_info, dict):
                 service_account_info = dict(service_account_payload)

            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            # Geçici dosya oluşturmaya gerek yok, doğrudan credentials kullanılabilir
            
            # Ortam değişkenlerini yine de ayarlayalım (bazı kütüphaneler için gerekebilir)
            os.environ["GOOGLE_PROJECT_ID"] = project_id
            # GOOGLE_APPLICATION_CREDENTIALS yerine doğrudan kimlik bilgisi kullanacağız
            
        except Exception as exc:
            st.error(f"Google servis hesabı JSON'u çözümlenemedi: {exc}")
            print(f"Service account parse hatası: {exc}")
            return None, None, None
    else:
        # Eğer payload yoksa, gcloud ADC'nin ayarlı olduğunu varsayıyoruz
        print("Servis hesabı JSON'u bulunamadı. Uygulama Varsayılan Kimlik Bilgileri (ADC) kullanılacak.")
        # Bu durumda credentials None kalır, kütüphaneler ADC'yi kullanır.

    os.environ["GOOGLE_LOCATION"] = location # Konumu ayarlayalım
    
    # VertexAI init'e gerek yok, kütüphaneler doğrudan credentials veya ADC kullanır
    st.session_state["location"] = location
    st.session_state["project_id"] = project_id # Proje ID'sini session state'e ekle
    st.session_state["embedding_model"] = DEFAULT_EMBEDDING_MODEL
    st.session_state["llm_model"] = DEFAULT_LLM_MODEL
    st.session_state["credentials_provided"] = bool(credentials)

    return project_id, location, credentials


# Uygulama başında kimlik bilgilerini yapılandır
PROJECT_ID, LOCATION, _CREDENTIALS = configure_google_credentials()


# --- Vektör Deposu ve Zincir Kurulum Fonksiyonları (ChromaDB için güncellendi) ---

def _get_vertex_credentials():
    """Service account varsa döner, aksi halde None (ADC)."""
    return _CREDENTIALS

def build_embeddings(project_id, location=None, model_name=DEFAULT_EMBEDDING_MODEL): # Model adını kontrol et
    """Vertex AI metin embedding modelini hazırlar."""
    # Kimlik bilgileri configure_google_credentials'dan alınacak (ADC veya service account)
    return VertexAIEmbeddings(
        project=project_id,
        location=location or LOCATION, # Global LOCATION kullan
        model_name=model_name,
        credentials=_get_vertex_credentials(),
    )

# @st.cache_resource kaldırıldı, çünkü load_vector_store artık oluşturma da yapabilir
def load_or_create_vector_store(project_id, persist_directory="chroma_db_recipes"):
    """Mevcut ChromaDB deposunu yükler, yoksa oluşturur."""
    if not project_id:
         st.error("Proje ID'si yüklenemedi.")
         return None
         
    embeddings = build_embeddings(project_id=project_id)
    
    if os.path.exists(persist_directory):
        try:
            st.write(f"Mevcut vektör deposu '{persist_directory}' yükleniyor...")
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            print("ChromaDB başarıyla yüklendi.")
            return vector_store
        except Exception as e:
            st.warning(f"Mevcut veritabanı yüklenemedi, yeniden oluşturulacak. Hata: {e}")
            # Hata durumunda yeniden oluşturmaya devam et
    
    # Eğer klasör yoksa veya yüklenemediyse, sıfırdan oluştur
    st.warning(f"'{persist_directory}' bulunamadı veya yüklenemedi. Veritabanı oluşturuluyor. Bu işlem birkaç dakika sürebilir...")
    recipe_docs = load_and_prepare_data('recipes.csv', sample_size=200) # sample_size'ı ayarla
    
    if recipe_docs:
        try:
            vector_store = Chroma.from_documents(
                documents=recipe_docs, 
                embedding=embeddings,
                persist_directory=persist_directory
            )
            vector_store.persist() # Kaydetmeyi unutma!
            st.success(f"'{persist_directory}' başarıyla oluşturuldu.")
            print(f"'{persist_directory}' başarıyla oluşturuldu.")
            return vector_store
        except Exception as e:
            st.error(f"Veritabanı oluşturulurken hata oluştu: {e}")
            print(f"Veritabanı oluşturulurken hata oluştu: {e}")
            return None
    else:
         st.error("Veri (recipes.csv) yüklenemediği için veritabanı oluşturulamadı.")
         print("Veri (recipes.csv) yüklenemediği için veritabanı oluşturulamadı.")
         return None


# @st.cache_resource kaldırıldı
def setup_conversational_chain(project_id, vector_store):
    """Sohbet zincirini oluşturur."""
    if not project_id or not vector_store:
        return None
    try:
        st.write("Sohbet zinciri oluşturuluyor...")
        llm = VertexAI(
            project=project_id,
            model_name=DEFAULT_LLM_MODEL,
            temperature=0.7,
            location=LOCATION,
            credentials=_get_vertex_credentials(),
        )
        
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer') # output_key eklendi
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            return_source_documents=True # Kaynak dokümanları da döndür (opsiyonel)
        )
        print("Sohbet zinciri başarıyla oluşturuldu.")
        return chain
    except Exception as e:
        st.error(f"Sohbet zinciri oluşturulurken bir hata oluştu: {e}")
        print(f"Sohbet zinciri oluşturulurken bir hata oluştu: {e}")
        return None

# --- Streamlit Arayüzü ---

st.set_page_config(page_title="Tarif Asistanı", layout="wide")
st.title("🍜 Sağlıklı Tarif Asistanı (RAG Chatbot)")
st.caption(
    f"LLM: {st.session_state.get('llm_model', DEFAULT_LLM_MODEL)} "
    f"/ Embedding: {st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL)} "
    f"/ Konum: {st.session_state.get('location', 'Bilinmiyor')}"
)

# Vektör deposunu ve RAG zincirini kur (state içinde sakla)
if "rag_chain" not in st.session_state:
    st.session_state.vector_store = load_or_create_vector_store(project_id=PROJECT_ID)
    if st.session_state.vector_store:
        st.session_state.rag_chain = setup_conversational_chain(project_id=PROJECT_ID, vector_store=st.session_state.vector_store)
    else:
        st.session_state.rag_chain = None

# Zincir başarıyla kurulduysa devam et
if st.session_state.rag_chain:
    
    # Sohbet geçmişini Streamlit session state'te tut
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # İlk karşılama mesajı
        st.session_state.messages.append({"role": "assistant", "content": "Merhaba! Ben sizin tarif asistanınızım. Nasıl bir tarif arıyorsunuz?"})

    # Geçmiş mesajları görüntüle
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan girdi al
    if prompt := st.chat_input("Tarif sorunuzu buraya yazın..."):
        
        # Kullanıcı mesajını geçmişe ekle ve göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistanın cevabını al
        with st.chat_message("assistant"):
            with st.spinner("Tarif aranıyor..."):
                try:
                    # LangChain zincirini çağırıyoruz
                    response = st.session_state.rag_chain.invoke({'question': prompt})
                    full_answer = response.get('answer', "Üzgünüm, bir cevap alamadım.") 
                    # Kaynakları göstermek isterseniz:
                    # source_docs = response.get('source_documents', [])
                    # if source_docs:
                    #     full_answer += "\n\n**Kaynaklar:**\n"
                    #     for doc in source_docs:
                    #          full_answer += f"- {doc.metadata.get('recipe_name', 'Bilinmeyen Tarif')}\n"
                             
                except Exception as e:
                    # Hata olursa logla ve kullanıcıya göster
                    print(f"Zincir çağrılırken hata: {e}")
                    st.error(f"Bir hata oluştu: {e}")
                    full_answer = "Üzgünüm, API çağrısında bir sorun oluştu. Lütfen uygulamanın loglarını kontrol edin."
                
                st.markdown(full_answer)
                
        # Asistan mesajını geçmişe ekle
        st.session_state.messages.append({"role": "assistant", "content": full_answer})

elif PROJECT_ID: # Eğer zincir kurulamadıysa ama Proje ID varsa
    st.error("Uygulama başlatılamadı. Lütfen logları kontrol edin veya veritabanının doğru oluşturulduğundan emin olun.")
