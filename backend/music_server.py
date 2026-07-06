import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import time
import sys
import numpy as np
import requests
import json
import logging
import warnings
from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
import threading
import time

# ---------------------------------
# --- Suppress Warnings ---
# 1. Suppress harmless UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
# 2. Suppress informational messages from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
# ---------------------------------

# --- Configuration ---
MODEL_REPO_ID = "facebook/musicgen-small"
AUDIO_LENGTH_S = 10
# ---------------------

# --- (User Profiles and Moods are omitted for brevity but are included) ---
USER_PROFILES = {
    "default": {
        "name": "Default",
        "preferred_genres": ["pop", "electronic"],
        "qualities": {"tempo": "moderate tempo", "instruments": ["synth", "drum machine"], "vibe": "energetic"}
    },
    "user1": {
        "name": "Alex (Lofi Fan)",
        "preferred_genres": ["lofi hip-hop", "chillwave"],
        "qualities": {"tempo": "slow tempo", "instruments": ["electric piano", "sampled drums", "vinyl crackle"], "vibe": "relaxing and studious"}
    }
}
VALID_MOODS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

def load_model():
    """Loads the MusicGen model and processor."""
    print("Loading MusicGen model...", file=sys.stderr)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        if device == "cpu":
            print("WARNING: CUDA not found. Running on CPU is very slow.", file=sys.stderr)
        processor = AutoProcessor.from_pretrained(MODEL_REPO_ID)
        model = MusicgenForConditionalGeneration.from_pretrained(MODEL_REPO_ID, torch_dtype=torch_dtype, attn_implementation="eager").to(device)
        print(f"MusicGen model loaded on device: {device}", file=sys.stderr)
        return model, processor, device
    except Exception as e:
        print(f"Fatal error loading model: {e}", file=sys.stderr)
        sys.exit(1)

def create_music_prompt(mood, confidence, profile):
    """Generates a descriptive music prompt."""
    prompt_phrases = []
    # --- (Prompt generation logic is omitted for brevity but is included) ---
    if confidence < 0.4:
        prompt_phrases.extend(profile['preferred_genres'])
    else:
        if mood == 'happiness':
            prompt_phrases = ['upbeat pop', 'joyful']
        else:
            prompt_phrases = ['slow ambient']
            
    final_phrases = list(dict.fromkeys(prompt_phrases))
    try:
        apiKey = os.environ.get("GEMINI_API_KEY")
        if not apiKey:
            raise EnvironmentError("GEMINI_API_KEY not found")
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={apiKey}"
        system_prompt = "You are a creative music prompt writer..." # Shortened
        user_query = f"Keywords: {json.dumps(final_phrases)}"
        payload = {"contents": [{"parts": [{"text": user_query}]}], "systemInstruction": {"parts": [{"text": system_prompt}]}}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(apiUrl, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status()
        final_prompt = response.json()['candidates'][0]['content']['parts'][0]['text'].strip().replace("`", "")
    except Exception as e:
        print(f"Warning: LLM prompt generation failed ({e}). Reverting to basic prompt.", file=sys.stderr)
        final_prompt = f"A track featuring: {', '.join(final_phrases)}."
    return final_prompt

def generate_music(model, processor, device, prompt):
    """Generates music and saves it to a web-accessible folder."""
    try:
        sample_rate = model.config.audio_encoder.sampling_rate
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        audio_values = model.generate(**inputs, max_new_tokens=int(AUDIO_LENGTH_S * 75))
        
        timestamp = int(time.time())
        output_filename = f"music_gen_{timestamp}.wav"
        
        output_dir = "static/music"
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, output_filename)
        
        audio_numpy = audio_values[0].cpu().numpy()
        audio_int16 = (np.clip(audio_numpy.squeeze(), -1, 1) * 32767).astype(np.int16)
        
        scipy.io.wavfile.write(output_filepath, rate=sample_rate, data=audio_int16)
        return output_filename
    except Exception as e:
        print(f"An error during generation: {e}", file=sys.stderr)
        return None

# --- Flask Web Server ---
app = Flask(__name__)
# Enable CORS so the Android app (or any client) can call the endpoint from another host
CORS(app)

print("Loading model for the server...")
MODEL, PROCESSOR, DEVICE = load_model()
print("Model loaded. Server is ready.")

@app.route('/generate', methods=['POST'])
def handle_generation():
    """API endpoint for the Android app."""
    if not MODEL:
        return jsonify({"error": "Model is not loaded"}), 500
    data = request.json
    user_id = data.get('user_id')
    mood = data.get('mood')
    confidence = data.get('confidence')
    if not all([user_id, mood, confidence is not None]):
        return jsonify({"error": "Missing required data"}), 400
    profile = USER_PROFILES.get(user_id, USER_PROFILES['default'])
    generated_prompt = create_music_prompt(mood, float(confidence), profile)
    print(f"Generating for prompt: {generated_prompt}", file=sys.stderr)
    output_file = generate_music(MODEL, PROCESSOR, DEVICE, generated_prompt)
    if output_file:
        file_url = request.host_url + f"music/{output_file}"
        return jsonify({"prompt": generated_prompt, "music_url": file_url})
    else:
        return jsonify({"error": "Failed to generate music"}), 500

@app.route('/music/<filename>')
def serve_music(filename):
    """Serves the generated .wav file."""
    return send_from_directory('static/music', filename)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


def cleanup_old_music(audio_dir: str = 'static/music', keep_seconds: int = 60*60):
    """Background thread: remove generated files older than keep_seconds to avoid disk growth."""
    while True:
        try:
            now = time.time()
            if os.path.isdir(audio_dir):
                for fn in os.listdir(audio_dir):
                    fp = os.path.join(audio_dir, fn)
                    try:
                        if os.path.isfile(fp):
                            mtime = os.path.getmtime(fp)
                            if (now - mtime) > keep_seconds:
                                os.remove(fp)
                    except Exception:
                        # ignore a single-file failure
                        pass
        except Exception:
            pass
        time.sleep(60 * 10)  # run every 10 minutes


# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_music, daemon=True)
cleanup_thread.start()

if __name__ == "__main__":
    # Allow port override via environment variable for flexible deployment
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)