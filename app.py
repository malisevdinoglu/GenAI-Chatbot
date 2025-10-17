import streamlit as st
import os

# main.py dosyasından gerekli fonksiyonları ve değişkeni import ediyoruz
# Not: main.py dosyanızın aynı klasörde olduğundan emin olun!
from main import load_vector_store, create_conversational_chain, PROJECT_ID 

# Streamlit, bu fonksiyonu sadece bir kere çalıştırır ve sonucunu önbelleğe alır.
@st.cache_resource
def setup_rag_pipeline():
    """RAG zincirini yükler ve hazırlar."""
    
    # Proje ID'sinin ayarlı olduğundan emin olalım (main.py dosyanızdan geliyor)
    if not PROJECT_ID or PROJECT_ID == "SENIN-PROJE-IDN":
        st.error("HATA: Lütfen main.py dosyasında geçerli bir Proje ID'si girin.")
        return None

    # Vektör deposu yükleniyor
    st.write("Vektör deposu yükleniyor...")
    vector_store = load_vector_store(project_id=PROJECT_ID)
    if not vector_store:
        st.error("Vektör deposu yüklenemedi. Lütfen FAISS klasörünün varlığını ve main.py'deki konumunu kontrol edin.")
        return None
    
    # Sohbet zinciri oluşturuluyor
    st.write("Sohbet zinciri oluşturuluyor...")
    chain = create_conversational_chain(project_id=PROJECT_ID, vector_store=vector_store)
    
    if not chain:
        st.error("Sohbet zinciri oluşturulurken bir hata oluştu.")
        return None
        
    return chain

# --- Streamlit Arayüzü ---

st.set_page_config(page_title="Tarif Asistanı", layout="wide")
st.title("🍜 Sağlıklı Tarif Asistanı (RAG Chatbot)")
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
                    full_answer = "Üzgünüm, API çağrısında bir sorun oluştu. Lütfen terminali kontrol edin."
                
                st.markdown(full_answer)
                
        # Asistan mesajını geçmişe ekle
        st.session_state.messages.append({"role": "assistant", "content": full_answer})