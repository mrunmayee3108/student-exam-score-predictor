import streamlit as st
import pickle

with open("student_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
gender_enc = data["gender_enc"]
parental_edu_enc = data["parental_edu_enc"]
internet_enc = data["internet_enc"]
extracurr_enc = data["extracurr_enc"]
part_time_enc = data["part_time_enc"]
age_scaler = data["age_scaler"]
study_hr_scaler = data["study_hr_scaler"]
netflix_scaler = data["netflix_scaler"]
attendace_scaler = data["attendace_scaler"]
sleep_scaler = data["sleep_scaler"]
excercise_scaler = data["excercise_scaler"]
mental_health_scaler = data["mental_health_scaler"]
social_media_scaler = data["social_media_scaler"]

st.set_page_config(page_title="Student Performance Analyser", page_icon='📈', layout = 'wide')
st.title("Student Exam Score Predictor📖")

gender = st.selectbox("Gender", ["Male", "Female"])
parental_education = st.selectbox("Parental Education", ["High School", "Bachelor", "Master"])
internet_quality = st.selectbox("Internet Quality", ["Poor", "Average", "Good"])
extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
part_time = st.selectbox("Part-time Job", ["Yes", "No"])
age = st.number_input("Age", 10, 30, 20)
study_hours = st.slider("Study Hours per Day", 0.0, 10.0, 5.0)
netflix_hours = st.slider("Netflix Hours", 0.0, 6.0, 1.0)
attendance = st.slider("Attendance (%)", 50.0, 100.0, 85.0)
sleep = st.slider("Sleep Hours", 4.0, 10.0, 7.0)
exercise = st.slider("Exercise Hours", 0.0, 3.0, 1.0)
mental_health = st.slider("Mental Health Rating", 0.0, 10.0, 7.0)
social_media = st.slider("Social Media Hours", 0.0, 6.0, 2.0)

if st.button("Predict Exam Score"):

    gender_val = gender_enc.transform([gender])[0]
    parental_edu_val = parental_edu_enc.transform([parental_education])[0]
    internet_val = internet_enc.transform([internet_quality])[0]
    extracurr_val = extracurr_enc.transform([extracurricular])[0]
    part_time_val = part_time_enc.transform([part_time])[0]

    age_scaled = age_scaler.transform([[age]])[0][0]
    study_scaled = study_hr_scaler.transform([[study_hours]])[0][0]
    netflix_scaled = netflix_scaler.transform([[netflix_hours]])[0][0]
    attendance_scaled = attendace_scaler.transform([[attendance]])[0][0]
    sleep_scaled = sleep_scaler.transform([[sleep]])[0][0]
    exercise_scaled = excercise_scaler.transform([[exercise]])[0][0]
    mental_scaled = mental_health_scaler.transform([[mental_health]])[0][0]
    social_scaled = social_media_scaler.transform([[social_media]])[0][0]

    final_input = [[
        gender_val,
        parental_edu_val,
        internet_val,
        extracurr_val,
        part_time_val,
        age_scaled,
        study_scaled,
        netflix_scaled,
        attendance_scaled,
        sleep_scaled,
        exercise_scaled,
        mental_scaled,
        social_scaled
    ]]

    prediction = model.predict(final_input)[0]

    st.success(f"Predicted Exam Score: {round(prediction, 2)}")