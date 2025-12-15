import os
# Force Keras 2 compatibility for Transformers
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer


# --- CONFIGURATION ---
MODEL_PATH = r"e:\vs codes\7la2 els7a\transformer\final t5 model with loss"
MAX_LENGTH = 256

print("Loading BioT5 Model (Fixed Loss Version)...")
try:
    # Load from the local directory
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = TFT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    print("✅ Model loaded successfully from local path.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Fallback removed to avoid confusion
    raise e

def ask_biot5(question):
    # 1. Prepare Input
    input_text = "question: " + question
    
    # 2. Tokenize
    inputs = tokenizer(
        input_text, 
        return_tensors="tf", 
        max_length=MAX_LENGTH, 
        truncation=True, 
        padding="max_length"
    )
    
    # 3. Generate
    # print(f"DEBUG: Generating...") 
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=MAX_LENGTH,
        num_beams=4,       # Restore quality generation
        early_stopping=True,
        repetition_penalty=2.0
    )
    
    # 4. Decode
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# --- INTERACTIVE LOOP ---
if __name__ == "__main__":
    print("\n💉 BioT5 Medical Chatbot Ready! (Type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        try:
            print("BioT5: ...", end="\r")
            answer = ask_biot5(user_input)
            print(f"BioT5: {answer}")
        except Exception as e:
            print(f"Error: {e}")
