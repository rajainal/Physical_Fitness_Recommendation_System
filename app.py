import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://raw.githubusercontent.com/rajainal/deploy_streamlit/main/wallpaper.png");
background-size: cover;
}

[data-testid="stVerticalBlockBorderWrapper"] {
border-radius: 10px;
padding: 0 10px 10px 10px;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

def load_model():
    with open("BP_LightGBM.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()
scaler = StandardScaler()

data = pd.read_csv("https://raw.githubusercontent.com/rajainal/deploy_streamlit/main/BPClean_Dataset.csv")

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

def gym_workout_plan(classification):
    plans = {
        'Excellent': {
            'Frekuensi': '5-6 kali per minggu',
            'Durasi': '60-90 menit per sesi',
            'Kardio': '4-5 kali per minggu, 30-45 menit (pilih: lari, bersepeda, berenang)',
            'Angkat Beban': '4-5 kali per minggu, split routine',
            'Peregangan dan Mobilitas': 'Setiap sesi latihan, minimal 10-15 menit'
        },
        'Great': {
            'Frekuensi': '4-5 kali per minggu',
            'Durasi': '45-60 menit per sesi',
            'Kardio': '3-4 kali per minggu, 20-30 menit (pilih: jogging, aerobik)',
            'Angkat Beban': '3-4 kali per minggu, bisa full-body atau split routine ringan',
            'Latihan Fungsional': '1 kali per minggu, seperti circuit training atau bodyweight exercises',
            'Peregangan dan Mobilitas': 'Setiap sesi latihan, minimal 10-15 menit'
        },
        'Average': {
            'Frekuensi': '3-4 kali per minggu',
            'Durasi': '30-45 menit per sesi',
            'Kardio': '2-3 kali per minggu, 20-30 menit (pilih: jalan cepat, sepeda statis)',
            'Angkat Beban': '2-3 kali per minggu, full-body workout',
            'Latihan Fungsional': '1 kali per minggu, dengan intensitas rendah hingga menengah',
            'Peregangan dan Mobilitas': 'Setiap sesi latihan, minimal 10-15 menit'
        },
        'Poor': {
            'Frekuensi': '2-3 kali per minggu',
            'Durasi': '20-30 menit per sesi',
            'Kardio': '1-2 kali per minggu, 15-20 menit (pilih: jalan santai, sepeda santai)',
            'Angkat Beban': '1-2 kali per minggu, fokus pada gerakan dasar dengan berat badan atau beban ringan',
            'Latihan Fungsional': 'Latihan ringan seperti yoga atau pilates',
            'Peregangan dan Mobilitas': 'Setiap sesi latihan, minimal 10-15 menit'
        }
    }
    return plans[classification]

def food_recommendations(classification):
    foods = {
        'Excellent': {
            'Sarapan': 'Oatmeal dengan buah-buahan dan kacang-kacangan',
            'Makan Siang': 'Salad ayam dengan quinoa dan sayuran',
            'Makan Malam': 'Ikan panggang dengan brokoli dan ubi jalar',
            'Snack': 'Yogurt dengan buah segar'
        },
        'Great': {
            'Sarapan': 'Telur orak-arik dengan roti gandum dan alpukat',
            'Makan Siang': 'Sandwich kalkun dengan sayuran',
            'Makan Malam': 'Ayam panggang dengan nasi merah dan sayuran hijau',
            'Snack': 'Smoothie buah dengan protein powder'
        },
        'Average': {
            'Sarapan': 'Roti panggang dengan selai kacang dan pisang',
            'Makan Siang': 'Nasi dengan ayam dan sayuran',
            'Makan Malam': 'Dada ayam panggang dengan salad',
            'Snack': 'Buah potong seperti apel atau jeruk'
        },
        'Poor': {
            'Sarapan': 'Sereal gandum dengan susu',
            'Makan Siang': 'Sup sayuran dengan roti gandum',
            'Makan Malam': 'Pasta dengan saus tomat dan sayuran',
            'Snack': 'Kacang-kacangan atau buah-buahan kering'
        }
    }
    return foods[classification]

def preprocess_input(data):
    return scaler.transform(np.array(data).reshape(1, -1))

def get_classification(model, data):
    predictions = model.predict(data)
    class_mapping = {1: 'Excellent', 2: 'Great', 3: 'Average', 4: 'Poor'}
    return class_mapping[predictions[0]]

def display_workout_plan(classification):
    workout_plan = gym_workout_plan(classification)
    st.markdown(f"### Performa Tubuh: {classification}")

    for key, value in workout_plan.items():
        st.markdown(f"""
        <div style="background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
            <h4 style="color: #2E8B57;">{key}</h4>
            <p>{value}</p>
        </div>
        """, unsafe_allow_html=True)

def display_food_plan(classification):
    food_plan = food_recommendations(classification)
    st.markdown(f"### Rekomendasi Makanan untuk Kategori: {classification}")

    for key, value in food_plan.items():
        st.markdown(f"""
        <div style="background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
            <h4 style="color: #2E8B57;">{key}</h4>
            <p>{value}</p>
        </div>
        """, unsafe_allow_html=True)

#=============================================================================================================================

st.title("Physical Fitness Advisory System")

Gender = [
    "Select your gender",
    "Male",
    "Female"
]

with st.form("da form"):
    age = st.number_input("Usia*", min_value=20, max_value=65, value=20, step=1, format="%d")

    left, right = st.columns(2)

    with left:
        choose = st.selectbox("Gender*", options=Gender, index=0)
        height = st.number_input("Tinggi Badan (cm)*", min_value=140.0, max_value=200.0, value=170.0, step=0.01)
        weight = st.number_input("Massa Badan (kg)*", min_value=34.0, max_value=102.0, value=60.0, step=0.01)
        bodyFat = st.number_input("Lemak Tubuh*", min_value=3.0, max_value=50.0, value=20.0, step=0.01)
        diastolic = st.number_input("Tekanan Darah Diastolik (mmHg)*", min_value=48, max_value=110, value=70, step=1, format="%d")
    
    with right:
        systolic = st.number_input("Tekanan Darah Sistolik (mmHg)*", min_value=80, max_value=172, value=100, step=1, format="%d")
        gripForce = st.number_input("Kekuatan Genggaman (kg)*", min_value=0.0, max_value=80.0, value=50.0, step=0.01)
        sitBendFw = st.number_input("Duduk dan Membungkuk ke Depan (cm)*", min_value=0.0, max_value=35.0, value=20.0, step=0.01)
        sitUps = st.number_input("Sit Ups*", min_value=0, max_value=55, value=30, step=1, format="%d")
        broadJump = st.number_input("Lompat Jauh (cm)*", min_value=0.0, max_value=305.0, value=80.0, step=0.01)

    st.markdown("**required*")
    submit_button = st.form_submit_button("Rekomendasi Latihan Saya")

    if  choose == "Male":
        gender = 0
    elif choose == "Female":
        gender = 1

    if submit_button:
        if choose == "Select your gender":
            st.warning("Please select a valid gender")
            st.stop()
        else:
            userInput = [age, gender, height, weight, bodyFat, diastolic, systolic, gripForce, sitBendFw, sitUps, broadJump]
            dataScalling = preprocess_input(userInput)
            predicted_class = get_classification(model, dataScalling)
            display_workout_plan(predicted_class)
            display_food_plan(predicted_class)