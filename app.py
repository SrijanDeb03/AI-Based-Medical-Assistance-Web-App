from flask import Flask, request, jsonify, render_template, send_from_directory, session
import requests
import pickle
import pandas as pd
import numpy as np
import re
from itertools import combinations
from collections import Counter
import operator
import os
import sys
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import tensorflow as tf
import uuid

# NEW: Import the Gemini SDK
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Gemini API key for chatbot

# Configure Gemini SDK
if not GEMINI_API_KEY:
    raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "models/gemini-1.5-flash-latest"  # Use a supported Gemini model

# Load chest X-ray model
try:
    xray_model = load_model('data/chest_xray.h5')
except FileNotFoundError as e:
    print(f"Error: Chest X-ray model file not found! Details: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading chest X-ray model: {str(e)}")
    sys.exit(1)

# Load COVID-19 X-ray model
try:
    covid_model = load_model('data/covid_final.h5')
except FileNotFoundError as e:
    print(f"Error: COVID-19 model file not found! Details: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading COVID-19 model: {str(e)}")
    sys.exit(1)

# Load skin disease model
try:
    skin_model = load_model('data/trial_skin.h5')
except FileNotFoundError as e:
    print(f"Error: Skin disease model file not found! Details: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading skin disease model: {str(e)}")
    sys.exit(1)

# Class labels for X-ray diseases
class_names = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Class labels for COVID-19 X-ray
covid_classes = ["COVID-19", "Pneumonia", "Normal"]

# Full names of skin disease classes
skin_classes = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Skin Lesion"
}
skin_class_labels = list(skin_classes.keys())
SKIN_IMG_SIZE = (224, 224)  # From reference code

# Preprocessing function for images
def preprocess_image(image_path, target_size=(128, 128)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_xray(image_path, model, class_labels):
    try:
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_confidences = predictions[0][top_3_indices] * 100
        top_3_classes = [class_labels[i] for i in top_3_indices]
        return [{"disease": cls, "confidence": round(conf, 2)} for cls, conf in zip(top_3_classes, top_3_confidences)]
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def predict_covid_xray(image_path, model, class_labels, top_k=3):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)[0]
    temperature = 0.5
    scaled_preds = np.exp(predictions / temperature) / np.sum(np.exp(predictions / temperature))
    top_indices = np.argsort(scaled_preds)[-top_k:][::-1]
    top_labels = [class_labels[i] for i in top_indices]
    top_probs = [float(round(scaled_preds[i] * 100, 2)) for i in top_indices]
    return [{"disease": label, "confidence": round(prob, 2)} for label, prob in zip(top_labels, top_probs)]

def predict_skin(image_path, model, class_labels):
    try:
        img_array = preprocess_image(image_path, target_size=SKIN_IMG_SIZE)
        predictions = model.predict(img_array)
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_confidences = predictions[0][top_3_indices] * 100
        top_3_classes = [class_labels[i] for i in top_3_indices]
        return [{"disease": skin_classes[cls], "confidence": round(conf, 2)} for cls, conf in zip(top_3_classes, top_3_confidences)]
    except Exception as e:
        raise Exception(f"Skin prediction failed: {str(e)}")

# --- Gemini Caching and Session Management ---
GEMINI_CACHE = None
CHAT_SESSIONS = {}  # In-memory chat history keyed by session ID

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. "
    "Answer briefly and clearly. "
    "Format your response in Markdown. "
    "Provide suggestions or steps, use bold headings, bullet points, and warnings where appropriate. "
    "Answer prompts regarding only medical conditions and basic greetings, for others reply that you are a medical chat bot"
    "Always end with a medical disclaimer."
)

def create_gemini_cache():
    """Creates and caches the system prompt for the Gemini model."""
    global GEMINI_CACHE
    print("Attempting to create or retrieve Gemini cache for system prompt...")
    try:
        # This creates a cache with a 60-minute time-to-live.
        # If a compatible cache from a previous run exists, it may be reused.
        GEMINI_CACHE = genai.caching.create_cached_content(
            model=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            ttl="60m"
        )
        print(f"Gemini cache created successfully. Name: {GEMINI_CACHE.name}")
    except Exception as e:
        print(f"Could not create Gemini cache. Chat will proceed without it. Error: {e}")

# Create the cache on application startup
create_gemini_cache()
# --- End of Caching and Session Setup ---

# --- Gemini Medical Chatbot Functions (UPDATED) ---

def format_response_markdown_chatbot(text):
    disclaimer = (
        "⚠️ *This response is generated by an AI medical assistant and is for informational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.*"
    )
    import re
    # Remove any lines containing common disclaimer phrases (case-insensitive)
    text = re.sub(
        r"(?im)^.*(disclaimer:|not a substitute for professional medical advice|always consult a qualified healthcare provider).*$(\n)?",
        "",
        text
    ).strip()
    # Remove any duplicate warning blocks that may remain
    text = re.sub(
        r"⚠️\s*\*This response is generated by an AI medical assistant.*?regarding a medical condition\.\*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()
    # Append the disclaimer only once
    if disclaimer not in text:
        if not text.endswith("---"):
            text += "\n\n---\n"
        text += disclaimer
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/covid19')
def covid_xray():
    return render_template('covid19.html')

@app.route('/predict_xray', methods=['GET', 'POST'])
def predict_xray_route():
    if request.method == 'POST':
        if 'xray_image' not in request.files:
            return render_template('xray_result.html', error="No image uploaded")
        file = request.files['xray_image']
        if file.filename == '':
            return render_template('xray_result.html', error="No file selected")
        upload_folder = 'static/uploads'
        try:
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            predictions = predict_xray(file_path, xray_model, class_names)
            return render_template('xray_result.html', predictions=predictions, image_path=f'uploads/{file.filename}')
        except Exception as e:
            return render_template('xray_result.html', error=f"Analysis error: {str(e)}")
    return render_template('predict_xray.html')

@app.route('/predict_covid_xray', methods=['POST'])
def predict_covid_xray_route():
    if 'xray_image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['xray_image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    predictions = predict_covid_xray(file_path, covid_model, covid_classes)
    return jsonify({"predictions": predictions, "image_path": f'uploads/{file.filename}'})

@app.route('/predict_skin', methods=['GET', 'POST'])
def predict_skin_route():
    if request.method == 'POST':
        if 'skin_image' not in request.files:
            return render_template('skin_result.html', error="No image uploaded")
        file = request.files['skin_image']
        if file.filename == '':
            return render_template('skin_result.html', error="No file selected")
        upload_folder = 'static/uploads'
        try:
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            predictions = predict_skin(file_path, skin_model, skin_class_labels)
            return render_template('skin_result.html', predictions=predictions, image_path=f'uploads/{file.filename}')
        except Exception as e:
            return render_template('skin_result.html', error=f"Analysis error: {str(e)}")
    return render_template('predict_skin.html')

# --- Gemini Chatbot API Endpoint (updated) ---
@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get('user_input')

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']

    try:
        # If caching failed at startup, GEMINI_CACHE will be None
        if GEMINI_CACHE:
            model = genai.GenerativeModel.from_cached_content(cached_content=GEMINI_CACHE)
            generation_config = {"cached_content": GEMINI_CACHE.name}
        else:
            # Fallback to a non-cached model if cache creation failed
            model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
            generation_config = {} # No cache to reference

        if session_id not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_id] = model.start_chat(history=[])
        
        chat = CHAT_SESSIONS[session_id]
        response = chat.send_message(user_input, generation_config=generation_config)
        
        formatted_response = format_response_markdown_chatbot(response.text)
        return jsonify({'response': formatted_response})

    except Exception as e:
        print(f"Error during Gemini chat: {e}")
        return jsonify({'response': f"An error occurred while communicating with the AI assistant: {str(e)}"}), 500

# Optional: clean & attach AI disclaimer
def format_response_markdown(text):
    disclaimer = (
        "⚠️ *This response is generated by an AI medical assistant and is for informational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.*"
    )
    # Basic strip without aggressive filtering
    text = text.strip()
    if disclaimer not in text:
        if not text.endswith("---"):
            text += "\n\n---\n"
        text += disclaimer
    return text.strip()

# 1. Symptom suggestion route using LLM
@app.route('/suggest_symptoms', methods=['POST'])
def suggest_symptoms():
    data = request.get_json()
    user_symptoms = data.get('user_symptoms', [])
    prompt = (
        f"You are a medical expert assistant. Given the following symptoms: {', '.join(user_symptoms)}, "
        "suggest up to 5 similar or related symptoms that the user might want to consider. "
        "Return only the symptoms as a comma-separated list."
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    suggestions = [s.strip() for s in response.text.split(",") if s.strip()]
    return jsonify({'suggestions': suggestions})

# 2. Disease prediction route using LLM
@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    data = request.json
    final_symptoms = data.get("final_symptoms", [])
    if not final_symptoms:
        return jsonify({"error": "No symptoms provided."}), 400

    prompt = (
        "You are a medical expert assistant. Given the following symptoms: "
        f"{', '.join(final_symptoms)}\n"
        "Predict the top 5 possible diseases based on these symptoms, or respond with 'Cannot find anything'.\n"
        "Format your answer as a Markdown table with two columns: Number and Disease.\n"
        "Example format:\n"
        "| Number | Disease |\n"
        "|--------|---------|\n"
        "| 1      | Flu     |\n"
        "Only return the table."
    )

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)

    if "Cannot find anything" in response.text or not response.text.strip():
        return jsonify({'response': "⚠️ Could not find a relevant prediction for the given symptoms."})

    markdown = format_response_markdown(response.text)
    return jsonify({'response': markdown})

# 3. Disease info route using LLM
@app.route('/get_disease_info', methods=['POST'])
def get_disease_info():
    data = request.get_json()
    disease = data.get('disease', '')
    if not disease:
        return jsonify({'error': 'No disease provided.'}), 400

    prompt = (
        f"Give a brief, user-friendly summary of the disease '{disease}'. "
        "Include common symptoms, causes, risk factors, and when to see a doctor. "
        "Format the answer in Markdown with clear bold headings and bullet points, readable format."
        "Keep it concise (4-6 sentences)."
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    markdown = format_response_markdown(response.text)
    return jsonify({'response': markdown})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
