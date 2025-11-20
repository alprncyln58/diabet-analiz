import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Diyabet Risk Analizi", page_icon="🩺", layout="wide")

# --- BAŞLIK VE GİRİŞ ---
st.title("🩺 Yapay Zeka Destekli Diyabet Risk Hesaplayıcı")
st.markdown("""
Bu sistem, makine öğrenmesi (Decision Tree) kullanarak hastaların klinik verilerine göre 
**Tip 2 Diyabet** riskini öngörmek için tasarlanmıştır.
*Veri Kaynağı: Pima Indians Diabetes Database*
""")

# --- YAN PANEL (DOKTOR GİRİŞİ) ---
st.sidebar.header("📋 Hasta Verilerini Giriniz")

def user_input_features():
    gebelik = st.sidebar.slider('Gebelik Sayısı', 0, 15, 1)
    glikoz = st.sidebar.slider('Glikoz (OGTT)', 0, 200, 110)
    tansiyon = st.sidebar.slider('Kan Basıncı (Diyastolik)', 0, 122, 72)
    cilt = st.sidebar.slider('Cilt Kalınlığı (mm)', 0, 99, 25)
    insulin = st.sidebar.slider('İnsülin (mu U/ml)', 0, 846, 30)
    bmi = st.sidebar.slider('BMI (Vücut Kitle İndeksi)', 0.0, 67.0, 30.5)
    soyagaci = st.sidebar.slider('Diyabet Soyağacı Fonksiyonu', 0.078, 2.42, 0.37)
    yas = st.sidebar.slider('Yaş', 21, 81, 29)
    
    data = {
        'Gebelik': gebelik,
        'Glikoz': glikoz,
        'Tansiyon': tansiyon,
        'CiltKalinligi': cilt,
        'Insulin': insulin,
        'BMI': bmi,
        'Soyagaci': soyagaci,
        'Yas': yas
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- MODELİN ARKA PLANDA EĞİTİLMESİ ---
# Streamlit her tıklamada kodu baştan çalıştırır. 
# @st.cache_resource sayesinde modeli bir kere eğitip hafızada tutuyoruz (Hız sağlar).
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Gebelik', 'Glikoz', 'Tansiyon', 'CiltKalinligi', 'Insulin', 'BMI', 'Soyagaci', 'Yas', 'Sonuc']
    df = pd.read_csv(url, names=column_names)
    
    X = df.drop('Sonuc', axis=1)
    y = df['Sonuc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, X

model, accuracy, X = train_model()

# --- ANA EKRAN (SONUÇLAR) ---

st.subheader("1. Girilen Hasta Verileri")
st.write(input_df)

# Tahmin Yapma
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("2. Risk Analizi Sonucu")
col1, col2 = st.columns(2)

with col1:
    if prediction[0] == 1:
        st.error("⚠️ SONUÇ: YÜKSEK RİSK")
        st.markdown(f"Model, bu hastanın diyabet olma ihtimalini **%{prediction_proba[0][1]*100:.2f}** olarak hesapladı.")
    else:
        st.success("✅ SONUÇ: DÜŞÜK RİSK")
        st.markdown(f"Model, bu hastanın sağlıklı olma ihtimalini **%{prediction_proba[0][0]*100:.2f}** olarak hesapladı.")

with col2:
    st.info(f"ℹ️ Model Doğruluğu: %{accuracy*100:.2f}")
    st.caption("Bu model klinik karar vermek için değil, ön eleme için tasarlanmıştır.")

# --- GÖRSELLEŞTİRME (HOCALARIN SEVDİĞİ KISIM) ---
st.subheader("3. Model Kararını Etkileyen Faktörler")
st.markdown("Yapay zeka karar verirken hangi veriye daha çok önem verdi?")

# Özellik önemlerini görselleştirme
feature_importance = pd.DataFrame(model.feature_importances_,
                                index = X.columns,
                                columns=['Önem Derecesi']).sort_values('Önem Derecesi', ascending=False)

st.bar_chart(feature_importance)