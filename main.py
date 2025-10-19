import os
import pandas as pd
from langchain_core.documents import Document 
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Vertex AI'dan sadece LLM için gerekli olanı import ediyoruz
from langchain_google_vertexai import VertexAI 
# VertexAIEmbeddings importu kaldırıldı.

PROJECT_ID = "genai-final-project-475415" 

def load_vector_store(project_id, index_path="chroma_db_recipes"): 
    """Kaydedilmiş Chroma vektör deposunu yükler."""
    # Index path'i Chroma'ya ayarlandı.
    if not os.path.exists(index_path):
        print(f"Hata: '{index_path}' klasörü bulunamadı.") 
        return None
    try:
        # !!! KRİTİK DEĞİŞİKLİK: Embeddings objesi kaldırıldı !!!
        # ChromaDB kendi varsayılan (lokal) embeddings modelini kullanacak.
        vector_store = Chroma(persist_directory=index_path) 
        
        print("Vektör deposu başarıyla yüklendi.")
        return vector_store
    except Exception as e:
        print(f"Vektör deposu yüklenirken bir hata oluştu: {e}")
        return None

def create_conversational_chain(project_id, vector_store):
    """Sohbet zincirini oluşturur."""
    try:
        # LLM (Gemini) için Vertex AI kullanılmaya devam ediyor.
        llm = VertexAI(project=project_id, model_name="gemini-2.5-flash", temperature=0.7, location="us-central1")
        
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