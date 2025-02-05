from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')  

SYMPTOM_COLUMNS = [
    "itching", "weight_loss", "dark_urine", "excessive_hunger", "sweating", "loss_of_appetite", "skin_rash", "headache", "stomach_pain", "ulcers_on_tongue", "dehydration", "family_history", "mucoid_sputum", "extra_marital_contacts", "unsteadiness", "mood_swings", "malaise", "back_pain", "swelling_joints", "knee_pain", "indigestion", "pain_during_bowel_movements", "toxic_look_(typhos)", "throat_irritation", "shivering", "fatigue", "depression", "chills", "dizziness", "increased_appetite", "enlarged_thyroid", "yellowing_of_eyes", "puffy_face_and_eyes", "diarrhoea", "constipation", "internal_itching", "hip_joint_pain", "burning_micturition", "breathlessness", "redness_of_eyes", "mild_fever", "drying_and_tingling_lips", "irregular_sugar_level", "cold_hands_and_feets", "continuous_sneezing", "neck_pain", "passage_of_gases", "nausea", "sinus_pressure", "belly_pain", "weakness_of_one_body_side", "painful_walking", "spotting_ urination", "joint_pain", "muscle_weakness", "polyuria", "watering_from_eyes", "restlessness", "slurred_speech", "irritation_in_anus", "yellowish_skin", "bloody_stool", "pain_behind_the_eyes", "dischromic _patches", "swollen_extremeties", "abdominal_pain", "pain_in_anal_region", "loss_of_smell", "phlegm", "vomiting", "sunken_eyes", "blurred_and_distorted_vision", "acidity", "weakness_in_limbs", "anxiety", "muscle_pain", "red_spots_over_body", "congestion", "lethargy", "muscle_wasting", "obesity", "visual_disturbances", "brittle_nails", "spinning_movements", "high_fever", "lack_of_concentration", "chest_pain", "cough", "altered_sensorium", "irritability", "abnormal_menstruation", "depression", "patches_in_throat", "stiff_neck", "loss_of_balance", "swelled_lymph_nodes", "palpitations", "fast_heart_rate", "weight_gain", "runny_nose", "nodal_skin_eruptions", "blood_in_sputum"
]

@app.route('/')
def home():
    return render_template('index.html', symptoms=SYMPTOM_COLUMNS)

@app.route('/predict', methods=['POST'])
def predict():
   
    symptoms = request.form.getlist('symptoms')
    
    input_vector = np.zeros(len(SYMPTOM_COLUMNS))
    for symptom in symptoms:
        if symptom in SYMPTOM_COLUMNS:
            input_vector[SYMPTOM_COLUMNS.index(symptom)] = 1

    input_vector = input_vector.reshape(1, -1)
    input_vector_scaled = scaler.transform(input_vector)

    predicted_class = model.predict(input_vector_scaled)

    predicted_label = label_encoder.inverse_transform(predicted_class)

    return f"Predicted Disease: {predicted_label[0]} <br><a href='/'>Go back</a>"

if __name__ == '__main__':
    app.run(debug=True)
