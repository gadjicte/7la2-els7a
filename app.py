import os
# Force Keras 2 compatibility for Transformers
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from flask import Flask, render_template, request, jsonify
import sys

# --- Configuration ---
app = Flask(__name__)

# Path to your FINE-TUNED BioT5 model
MODEL_PATH = r"E:\vs codes\7la2 els7a\model"
MAX_LENGTH = 256

# Global Variables
model = None
tokenizer = None

def load_biot5_system():
    global model, tokenizer
    print("Loading BioT5 Medical Model...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        model = TFT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        print("✅ BioT5 Model Loaded Successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def generate_response(question):
    try:
        input_text = "question: " + question
        
        # Tokenize
        inputs = tokenizer(
            input_text, 
            return_tensors="tf", 
            max_length=MAX_LENGTH, 
            truncation=True, 
            padding="max_length"
        )
        
        # Generate (Using Beam Search for Quality)
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=2.0
        )
        
        # Decode
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Generation Error: {e}")
        return "I'm sorry, I encountered an error while processing your request."

# --- Routes ---

@app.route("/")
def home():
    # Assumes you have an index.html in a 'templates' folder
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    if not model:
        return jsonify({"response": "System Error: Model not loaded."})
        
    user_text = request.form.get("msg")
    if not user_text:
        return jsonify({"response": "Please say something."})
    
    bot_response = generate_response(user_text)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    if load_biot5_system():
        # Run Flask
        # debug=False is safer for TF/Threading issues usually, but True is ok for simple apps
        app.run(debug=True, port=5000, use_reloader=False) 
    else:
        print("Failed to start app because model could not be loaded.")
