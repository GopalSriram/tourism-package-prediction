import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(
    repo_id="GopalSriram/tourism-package-model",
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

st.title("🌴 Tourism Wellness Package Predictor")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

age                       = st.number_input("Age", min_value=18, max_value=100, value=35)
city_tier                 = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch         = st.number_input("Duration of Pitch (mins)", min_value=0, max_value=100, value=15)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
number_of_followups       = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
preferred_property_star   = st.selectbox("Preferred Property Star", [3, 4, 5])
number_of_trips           = st.number_input("Number of Trips", min_value=0, max_value=20, value=3)
passport                  = st.selectbox("Has Passport?", [0, 1])
pitch_satisfaction_score  = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car                   = st.selectbox("Owns a Car?", [0, 1])
number_of_children        = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
monthly_income            = st.number_input("Monthly Income", min_value=0, max_value=100000, value=30000)
type_of_contact           = st.selectbox("Type of Contact", [0, 1])
occupation                = st.selectbox("Occupation", [0, 1, 2, 3])
gender                    = st.selectbox("Gender", [0, 1])
marital_status            = st.selectbox("Marital Status", [0, 1, 2])
designation               = st.selectbox("Designation", [0, 1, 2, 3, 4])
product_pitched           = st.selectbox("Product Pitched", [0, 1, 2, 3, 4])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'Age': age,
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'NumberOfFollowups': number_of_followups,
        'PreferredPropertyStar': preferred_property_star,
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children,
        'MonthlyIncome': monthly_income,
        'TypeofContact': type_of_contact,
        'Occupation': occupation,
        'Gender': gender,
        'MaritalStatus': marital_status,
        'Designation': designation,
        'ProductPitched': product_pitched,
    }])

    # Auto-match to exact columns the model was trained on
    try:
        model_features = model[:-1].get_feature_names_out()
    except Exception:
        model_features = None

    if model_features is None:
        # Add any missing columns the model expects with default value 0
        try:
            model.predict(input_data)  # test first
        except ValueError as e:
            missing = str(e).split("{'")[1].split("'}")[0].split("', '")
            for col in missing:
                input_data[col] = 0

    try:
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("✅ Customer WILL purchase the Wellness Package!")
        else:
            st.error("❌ Customer will NOT purchase the Wellness Package.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
