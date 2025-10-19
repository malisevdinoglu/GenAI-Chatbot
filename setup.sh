#!/bin/bash

# recipes.csv dosyasının varlığını kontrol et
if [ ! -f "recipes.csv" ]; then
    echo "recipes.csv dosyası bulunamadı. Veritabanı oluşturulamaz."
    exit 1
fi

# FAISS klasörü mevcut değilse, create_vector_store.py'yi çalıştır
if [ ! -d "faiss_index_recipes" ]; then
    echo "FAISS veritabanı bulunamadı, şimdi oluşturuluyor..."
    # Bu, Service Account yetkisi ile çalışmalıdır.
    python create_vector_store.py
else
    echo "FAISS veritabanı zaten mevcut."
fi