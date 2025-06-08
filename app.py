import streamlit as st
import pandas as pd
from prediction_func import predict_dropout
from preprocessing_func import preprocess_input_data

st.set_page_config(page_title="Prediksi Dropout", layout="wide")
st.title("🎓 Prediksi Dropout Mahasiswa - Jaya Jaya Institut")

with st.form("dropout_form"):
    st.subheader("📋 Lengkapi Form Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age at Enrollment", 17, 65, 25)
        admission_grade = st.slider("Admission Grade", 0, 200, 150)
        scholarship = st.radio("Scholarship Holder", ["No", "Yes"])
        displaced = st.radio("Displaced Student", ["No", "Yes"])
        debtor = st.radio("Debtor", ["No", "Yes"])
        tuition_up_to_date = st.radio("Tuition Fees Up to Date", ["Yes", "No"])
        international = st.radio("International Student", ["No", "Yes"])
        educational_needs = st.radio("Educational Special Needs", ["No", "Yes"])
        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Widower",
                "Divorced",
                "Facto Union",
                "Legally Separated",
            ],
        )
        app_mode = st.selectbox(
            "Application Mode",
            [
                "1st phase - general",
                "Change of course",
                "Transfer",
                "Over 23 years old",
            ],
        )
        app_order = st.slider("Application Order (0 = 1st Choice)", 0, 9, 0)
        prev_qual = st.selectbox(
            "Previous Qualification",
            ["Secondary", "Bachelor", "Degree", "Master", "Tech course"],
        )
        prev_qual_grade = st.slider("Previous Qualification Grade", 0, 200, 150)
        nationality = st.selectbox(
            "Nationality",
            ["Portuguese", "Brazilian", "Cape Verdean", "Guinean", "Spanish", "Other"],
        )

    with col2:
        mother_qual = st.slider("Mother's Qualification (kode)", 0, 44, 1)
        father_qual = st.slider("Father's Qualification (kode)", 0, 44, 1)
        mother_occ = st.slider("Mother's Occupation (kode)", 0, 195, 1)
        father_occ = st.slider("Father's Occupation (kode)", 0, 195, 1)
        day_evening = st.radio("Attendance Time", ["Daytime", "Evening"])

        # Semester 1
        cred_1 = st.slider("Curricular Units 1st Sem Credited", 0, 20, 0)
        enrolled_1 = st.slider("1st Sem Enrolled", 0, 20, 5)
        eval_1 = st.slider("1st Sem Evaluated", 0, 20, 5)
        approved_1 = st.slider("1st Sem Approved", 0, 20, 5)
        grade_1 = st.slider("1st Sem Grade", 0.0, 20.0, 12.0)
        without_eval_1 = st.slider("1st Sem Without Evaluation", 0, 10, 0)

        # Semester 2
        cred_2 = st.slider("Curricular Units 2nd Sem Credited", 0, 20, 0)
        enrolled_2 = st.slider("2nd Sem Enrolled", 0, 20, 5)
        eval_2 = st.slider("2nd Sem Evaluated", 0, 20, 5)
        approved_2 = st.slider("2nd Sem Approved", 0, 20, 5)
        grade_2 = st.slider("2nd Sem Grade", 0.0, 20.0, 12.0)
        without_eval_2 = st.slider("2nd Sem Without Evaluation", 0, 10, 0)

        # Ekonomi
        unemployment = st.slider("Unemployment Rate", 0.0, 20.0, 6.0)
        inflation = st.slider("Inflation Rate", 0.0, 20.0, 1.5)
        gdp = st.slider("GDP (juta euro)", 0.0, 100000.0, 50000.0)

    submitted = st.form_submit_button("🔍 Prediksi Dropout")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Age_at_enrollment": age,
                "Admission_grade": admission_grade,
                "Scholarship_holder": scholarship,
                "Displaced": displaced,
                "Debtor": debtor,
                "Tuition_fees_up_to_date": tuition_up_to_date,
                "International": international,
                "Educational_special_needs": educational_needs,
                "Marital_status": marital_status,
                "Application_mode": app_mode,
                "Application_order": app_order,
                "Previous_qualification": prev_qual,
                "Previous_qualification_grade": prev_qual_grade,
                "Nacionality": nationality,
                "Mothers_qualification": mother_qual,
                "Fathers_qualification": father_qual,
                "Mothers_occupation": mother_occ,
                "Fathers_occupation": father_occ,
                "Daytime_evening_attendance": day_evening,
                "Curricular_units_1st_sem_credited": cred_1,
                "Curricular_units_1st_sem_enrolled": enrolled_1,
                "Curricular_units_1st_sem_evaluations": eval_1,
                "Curricular_units_1st_sem_approved": approved_1,
                "Curricular_units_1st_sem_grade": grade_1,
                "Curricular_units_1st_sem_without_evaluations": without_eval_1,
                "Curricular_units_2nd_sem_credited": cred_2,
                "Curricular_units_2nd_sem_enrolled": enrolled_2,
                "Curricular_units_2nd_sem_evaluations": eval_2,
                "Curricular_units_2nd_sem_approved": approved_2,
                "Curricular_units_2nd_sem_grade": grade_2,
                "Curricular_units_2nd_sem_without_evaluations": without_eval_2,
                "Unemployment_rate": unemployment,
                "Inflation_rate": inflation,
                "GDP": gdp,
            }
        ]
    )

    preprocessed = preprocess_input_data(input_data)
    pred, prob = predict_dropout(preprocessed)

    st.markdown(f"## 🎓 Prediksi: **{'Dropout' if pred[0]==1 else 'Tidak Dropout'}**")
    st.metric("Probabilitas Dropout", f"{prob[0]*100:.2f}%")
