import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Pratik Diyabet Risk Analizi", page_icon="🩺", layout="wide")

# --- BAŞLIK ---
st.title("🩺 Pratik Diyabet Risk Hesaplayıcı")
st.markdown("""
Bu sistem, poliklinik şartlarında kolayca elde edilebilen verilerle (Kan tahlili, Tansiyon, BMI)
**Tip 2 Diyabet** riskini öngörmek için tasarlanmıştır.
*Gereksiz parametreler (Cilt kalınlığı vb.) çıkarılarak klinik kullanıma uygun hale getirilmiştir.*
""")

# --- YAN PANEL (DOKTOR GİRİŞİ) ---
st.sidebar.header("📋 Hasta Bulguları")

def user_input_features():
    # Artık sadece 6 parametre var
    gebelik = st.sidebar.slider('Gebelik Sayısı', 0, 15, 0)
    glikoz = st.sidebar.slider('Glikoz (OGTT - mg/dl)', 0, 200, 100)
    tansiyon = st.sidebar.slider('Kan Basıncı (Diyastolik - mmHg)', 0, 122, 70)
    # Cilt kalınlığı kaldırıldı
    insulin = st.sidebar.slider('İnsülin (mu U/ml)', 0, 846, 30)
    bmi = st.sidebar.slider('BMI (Vücut Kitle İndeksi)', 0.0, 67.0, 25.0)
    # Soyağacı fonksiyonu kaldırıldı
    yas = st.sidebar.slider('Yaş', 21, 81, 30)
    
    data = {
        'Gebelik': gebelik,
        'Glikoz': glikoz,
        'Tansiyon': tansiyon,
        'Insulin': insulin,
        'BMI': bmi,
        'Yas': yas
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- MODELİN EĞİTİLMESİ ---
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    # İsimlendirmeyi yapıyoruz
    column_names = ['Gebelik', 'Glikoz', 'Tansiyon', 'CiltKalinligi', 'Insulin', 'BMI', 'Soyagaci', 'Yas', 'Sonuc']
    df = pd.read_csv(url, names=column_names)
    
    # KRİTİK NOKTA: Kullanmayacağımız sütunları veri setinden atıyoruz
    df = df.drop(['CiltKalinligi', 'Soyagaci'], axis=1)
    
    X = df.drop('Sonuc', axis=1)
    y = df['Sonuc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modeli eğitiyoruz
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, X

model, accuracy, X = train_model()

# --- SONUÇ EKRANI ---

st.subheader("1. Girilen Klinik Veriler")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("2. Analiz Sonucu")
col1, col2 = st.columns(2)

with col1:
    if prediction[0] == 1:
        st.error("⚠️ TAHMİN: YÜKSEK RİSK")
        st.markdown(f"Algoritma, hastanın diyabet profiline **%{prediction_proba[0][1]*100:.2f}** oranında uyduğunu saptadı.")
    else:
        st.success("✅ TAHMİN: DÜŞÜK RİSK")
        st.markdown(f"Algoritma, hastanın sağlıklı profiline **%{prediction_proba[0][0]*100:.2f}** oranında uyduğunu saptadı.")

with col2:
    st.info(f"ℹ️ Model Doğruluğu: %{accuracy*100:.2f}")
    # Doğruluk oranı biraz düşebilir çünkü veri azalttık, bu normaldir.
    st.caption("Not: Parametre sayısı azaltıldığı için model sadece temel risk faktörlerine odaklanmaktadır.")

# --- GÖRSELLEŞTİRME ---
st.subheader("3. En Önemli Risk Faktörleri")
feature_importance = pd.DataFrame(model.feature_importances_,
                                index = X.columns,
                                columns=['Önem Derecesi']).sort_values('Önem Derecesi', ascending=False)

st.bar_chart(feature_importance)
