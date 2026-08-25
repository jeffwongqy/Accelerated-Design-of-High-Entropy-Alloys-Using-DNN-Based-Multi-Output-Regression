import streamlit as st
import pandas as pd
import joblib 
from sklearn.preprocessing import StandardScaler

# import data and scaler
model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("⚛️ High-Entropy Alloy Multi-Output Prediction")
st.caption("Machine Learning-Based Prediction of Mechanical Properties")

col1, col2, col3 = st.columns(3)

with col1:
    cr = st.number_input("Cr (%):", min_value = 0.00, max_value = 26.0)
    homo_temp = st.number_input("Homogenous Temperature (K):", min_value = 1273.00, max_value = 1523.00)
    cold_roll = st.number_input("Cold Rolling (%):", min_value = 0.00, max_value = 96.00)
    anneal_temp = st.number_input("Anneal Temperature (K):", min_value = 0.00, max_value = 1473.00)
    anneal_time = st.number_input("Anneal Time (H):", min_value = 0.00, max_value = 1.00)
    ape_mean = st.number_input("APE Mean:", min_value = 0.00, max_value = 0.006388)

with col2:
    radii_gamma = st.number_input("Radii Gamma:", min_value = 1.02, max_value = 1.08)
    vec_mean = st.number_input("VEC Mean:", min_value = 7.55, max_value = 8.45)
    mean_atomic_mass = st.number_input("Mean Atomic Mass:", min_value = 53.39, max_value = 56.59)
    constraint_1 = st.number_input("Constraint 1:", min_value = 0.00, max_value = 1173.00)
    constraint_2 = st.number_input("Constraint 2:", min_value = 0.00, max_value = 1173.00)

input_data = pd.DataFrame({'Cr': [cr], 
                          'Hom_Temp(K)': [homo_temp], 
                          'CR(%)': [cold_roll], 
                          'Anneal_Temp(K)': [anneal_temp],
                          'Anneal_Time(h)': [anneal_time],
                          'APE mean': [ape_mean], 
                          'Radii gamma': [radii_gamma], 
                          'VEC mean': [vec_mean], 
                          'mean atomic_mass': [mean_atomic_mass], 
                          'constraint_1': [constraint_1], 
                          'constraint_2': [constraint_2]})

input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    
    st.success("Prediction Completed!")
    
    st.write("Yield Strength: {:.2f} MPa".format(prediction[0][0]))
    st.write("Ultimate Tensile Strength: {:.2f} MPa".format(prediction[0][1]))
    st.write("Elongation: {:.2f}%".format(prediction[0][2]))