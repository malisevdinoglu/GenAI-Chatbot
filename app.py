import json
import os
from pathlib import Path

import streamlit as st
from google.oauth2 import service_account
import vertexai

from create_vector_store import load_and_prepare_data, create_new_vector_store
# main.py dosyasından gerekli fonksiyonları ve değişkeni import ediyoruz
from main import load_vector_store, create_conversational_chain, PROJECT_ID


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
    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    location = os.environ.get("GOOGLE_LOCATION", "us-central1")
    service_account_payload = (
        os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        or os.environ.get("GOOGLE_SERVICE_ACCOUNT")
    )

    # Streamlit secrets varsa yedek olarak kullan
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

    if not project_id or not service_account_payload:
        warning_msg = (
            "Google Cloud secrets yüklenemedi. Deploy ortamı için Streamlit secrets'ı "
            "veya ortam değişkenlerini (GOOGLE_PROJECT_ID, GOOGLE_SERVICE_ACCOUNT_JSON) ayarlayın."
        )
        st.warning(warning_msg)
        print("Secrets uyarısı: PROJECT_ID veya service account bulunamadı.")
        return None, None, None

    try:
        if isinstance(service_account_payload, str):
            service_account_info = json.loads(service_account_payload.strip())
        else:
            service_account_info = dict(service_account_payload)
    except Exception as exc:
        st.error("Google servis hesabı JSON'u çözümlenemedi. Ortam değişkenini/secrets'ı kontrol edin.")
        print(f"Service account parse hatası: {exc}")
        return None, None, None

    if not service_account_info:
        st.error("Servis hesabı bilgisi boş geldi. Ortam değişkenlerini kontrol edin.")
        return None, None, None

    if not isinstance(service_account_info, dict):
        service_account_info = dict(service_account_payload)

    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    credentials_path = Path("/tmp/google_service_account.json")
    credentials_path.write_text(json.dumps(service_account_info))

    os.environ["GOOGLE_PROJECT_ID"] = project_id
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    os.environ["GOOGLE_LOCATION"] = location

    vertexai.init(project=project_id, location=location, credentials=credentials)
    st.session_state["location"] = location

    return project_id, location, credentials


PROJECT_ID, LOCATION, _CREDENTIALS = configure_google_credentials()
# ---------------------------------------------------------------------------------


# Streamlit, bu fonksiyonu sadece bir kere çalıştırır ve sonucunu önbelleğe alır.

@st.cache_resource
def setup_rag_pipeline():
    """RAG zincirini yükler ve hazırlar, yoksa oluşturur."""
    
    # ChromaDB klasör adını tanımla
    CHROMA_DB_PATH = "chroma_db_recipes" 
    
    if not PROJECT_ID:
        st.error("HATA: Proje ID'si ayarlanmadı. Lütfen Streamlit Secrets'ı kontrol edin.")
        return None

    # 1. Vektör deposu yükleniyor
    st.write("Vektör deposu yükleniyor...")
    # Chroma'yı yüklemek için doğru klasör adını main.py'deki fonksiyona iletiyoruz.
    vector_store = load_vector_store(project_id=PROJECT_ID, index_path=CHROMA_DB_PATH)
    
    # 2. Eğer yüklenemezse (klasör yoksa), sıfırdan oluşturmayı dene (OTOMATİK OLUŞTURMA)
    if not vector_store:
        # Sarı uyarıyı gösterir (FAISS/Chroma'nın oluşturulduğu an)
        st.warning("Veritabanı bulunamadı. Veritabanı yeniden oluşturuluyor. Bu işlem birkaç dakika sürebilir...")
        
        # Veri setini yükle
        recipe_docs = load_and_prepare_data('recipes.csv') 
        
        if recipe_docs:
            # Vektör veritabanını oluştur ve kaydet
            create_new_vector_store(recipe_docs, PROJECT_ID)
            
            # Oluşturulduktan sonra tekrar yüklemeyi dene
            vector_store = load_vector_store(project_id=PROJECT_ID, index_path=CHROMA_DB_PATH)
            
            if not vector_store:
                 st.error("Veritabanı oluşturulduktan sonra bile yüklenemedi. Lütfen logları kontrol edin.")
                 return None
        else:
             # recipes.csv bulunamazsa bu hatayı verir.
             st.error("Veri (recipes.csv) yüklenemediği için veritabanı oluşturulamadı.")
             return None

    # 3. Sohbet zinciri oluşturuluyor
    st.write("Sohbet zinciri oluşturuluyor...")
    chain = create_conversational_chain(project_id=PROJECT_ID, vector_store=vector_store)
    
    if not chain:
        st.error("Sohbet zinciri oluşturulurken bir hata oluştu.")
        return None
        
    return chain

# --- Streamlit Arayüzü ---

st.set_page_config(page_title="Tarif Asistanı", layout="wide")
st.title("🍜 Sağlıklı Tarif Asistanı (RAG Chatbot)")
# Başlıkta location bilgisini de gösterelim
st.caption(f"LLM: Gemini / Embedding: text-embedding-004 / Konum: {st.session_state.get('location', 'us-central1')}")

# RAG zincirini kur
rag_chain = setup_rag_pipeline()

if rag_chain:
    
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
                    response = rag_chain.invoke({'question': prompt})
                    full_answer = response['answer']
                except Exception as e:
                    # Hata olursa logla ve kullanıcıya göster
                    print(f"Zincir çağrılırken hata: {e}")
                    full_answer = "Üzgünüm, API çağrısında bir sorun oluştu. Lütfen uygulamanın loglarını kontrol edin."
                
                st.markdown(full_answer)
                
        # Asistan mesajını geçmişe ekle
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
