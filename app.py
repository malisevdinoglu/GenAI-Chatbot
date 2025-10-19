import streamlit as st
import os

from create_vector_store import load_and_prepare_data, create_new_vector_store
# main.py dosyasından gerekli fonksiyonları ve değişkeni import ediyoruz
from main import load_vector_store, create_conversational_chain, PROJECT_ID 

# --- GÜNCELLENEN KISIM: Secrets'ı okuma ve Ortam Değişkeni olarak ayarlama ---
try:
    # Secrets'tan değişkenleri okuyup os.environ'a ayarlıyoruz.
    os.environ["GOOGLE_PROJECT_ID"] = st.secrets["GOOGLE_PROJECT_ID"] 
    os.environ["GOOGLE_SERVICE_ACCOUNT"] = st.secrets["GOOGLE_SERVICE_ACCOUNT"]

    PROJECT_ID = st.secrets["GOOGLE_PROJECT_ID"]

    st.session_state["location"] = "us-central1" # Streamlit başlığı için konumu sakla
except Exception:
    pass 
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