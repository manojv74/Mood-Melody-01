import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import time
import sys
import numpy as np
import requests  # <-- Added for API calls
import json      # <-- Added for API calls

# --- HIDE WARNINGS (NEW SECTION) ---
import logging
import warnings

# 1. Suppress the harmless "torch.tensor(sourceTensor)" UserWarning
# We're ignoring all UserWarnings, as they are not critical.
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Suppress the long "Config of the..." informational messages
# This tells the 'transformers' library to only log a message if it's a real ERROR.
logging.getLogger("transformers").setLevel(logging.ERROR)
# ---------------------------------

# --- Configuration ---
MODEL_REPO_ID = "facebook/musicgen-small" # Small model for max speed
AUDIO_LENGTH_S = 60 # Length of audio to generate in seconds
# ---------------------

# --- User Profile Database ---
USER_PROFILES = {
    "default": {
        "name": "Default",
        "preferred_genres": ["pop", "electronic"],
        "qualities": {
            "tempo": "moderate tempo",
            "instruments": ["synth", "drum machine"],
            "vibe": "energetic"
        }
    },
    "user1": {
        "name": "Alex (Lofi Fan)",
        "preferred_genres": ["lofi hip-hop", "chillwave"],
        "qualities": {
            "tempo": "slow tempo",
            "instruments": ["electric piano", "sampled drums", "vinyl crackle"],
            "vibe": "relaxing and studious"
        }
    },
    "user2": {
        "name": "Ben (Rocker)",
        "preferred_genres": ["classic rock", "alternative rock"],
        "qualities": {
            "tempo": "fast tempo",
            "instruments": ["electric guitar", "bass guitar", "acoustic drums"],
            "vibe": "driving and powerful"
        }
    },
    "user3": {
        "name": "Cara (Classical)",
        "preferred_genres": ["cinematic orchestral", "classical piano"],
        "qualities": {
            "tempo": "dynamic tempo",
            "instruments": ["string section", "piano", "brass"],
            "vibe": "majestic and emotional"
        }
    },
    "user4": {
        "name": "Dana (Ambient)",
        "preferred_genres": ["dark ambient", "drone", "soundscape"],
        "qualities": {
            "tempo": "very slow tempo",
            "instruments": ["evolving pads", "deep drone", "field recordings"],
            "vibe": "eerie and atmospheric"
        }
    }
}

VALID_MOODS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
# ---------------------------

def load_model():
    """
    Loads the MusicGen model, processor, and moves them to the active device.
    This is the slow, one-time setup.
    """
    print("Loading MusicGen controller model... (This may take a few minutes)", file=sys.stderr)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        if device == "cpu":
            print("WARNING: CUDA not found. Running on CPU. This will be VERY slow.", file=sys.stderr)

        processor = AutoProcessor.from_pretrained(MODEL_REPO_ID)
        model = MusicgenForConditionalGeneration.from_pretrained(
            MODEL_REPO_ID, 
            torch_dtype=torch_dtype,
            attn_implementation="eager"
        )
        model = model.to(device)
        
        print(f"MusicGen model loaded successfully on device: {device} (using {torch_dtype})", file=sys.stderr)
        return model, processor, device

    except Exception as e:
        print(f"Fatal error loading model: {e}", file=sys.stderr)
        sys.exit(1)

def create_music_prompt(mood, confidence, profile):
    """
    Intelligently generates a descriptive music prompt based on controller rules.
    """
    prompt_phrases = []

    # --- Rule 1: Confidence & Mood/Profile Logic ---
    if confidence < 0.4:
        # Prioritize user profile
        prompt_phrases.extend(profile['preferred_genres'])
        prompt_phrases.append(profile['qualities']['tempo'])
        prompt_phrases.extend(profile['qualities']['instruments'])
        prompt_phrases.append(profile['qualities']['vibe'])
        
    elif 0.4 <= confidence <= 0.7:
        # Blend mood and profile
        prompt_phrases.extend(profile['preferred_genres'])
        prompt_phrases.extend(profile['qualities']['instruments'])
        
        # Add mood elements
        if mood == 'happiness':
            prompt_phrases.append('uplifting')
            prompt_phrases.append('major key')
        elif mood == 'sadness':
            prompt_phrases.append('somber')
            prompt_phrases.append('reflective')
            prompt_phrases.append('slow tempo')
        elif mood == 'anger':
            prompt_phrases.append('driving')
            prompt_phrases.append('intense')
        elif mood == 'surprise':
            prompt_phrases.append('dynamic')
            prompt_phrases.append('unexpected shifts')
        else: # Neutral, Fear, Disgust, Contempt
            prompt_phrases.append(profile['qualities']['vibe']) # Default to profile vibe

    else: # confidence > 0.7
        # Strongly reflect mood
        if mood == 'happiness':
            prompt_phrases = ['upbeat pop', 'fast tempo', 'bright synths', 'acoustic guitar', 'joyful', 'euphoric']
        elif mood == 'sadness':
            prompt_phrases = ['slow ambient', 'solitary piano', 'string pads', 'melancholic', 'poignant', 'reflective']
        elif mood == 'anger':
            prompt_phrases = ['industrial rock', 'fast driving beat', 'distorted electric guitars', 'aggressive', 'intense']
        elif mood == 'surprise':
            prompt_phrases = ['cinematic score', 'sudden orchestral swells', 'dynamic percussion', 'unpredictable melody']
        elif mood == 'fear':
            prompt_phrases = ['dark ambient', 'dissonant strings', 'low drone', 'tense', 'eerie atmosphere']
        else: # Neutral, Disgust, Contempt
            prompt_phrases.extend(profile['preferred_genres'])
            prompt_phrases.append(profile['qualities']['vibe'])
            prompt_phrases.append('mellow tempo')

    # --- Build Final Prompt ---
    # Remove duplicates while preserving order
    final_phrases = []
    seen = set()
    for phrase in prompt_phrases:
        if phrase not in seen:
            final_phrases.append(phrase)
            seen.add(phrase)
            
    # --- *** NEW LLM PROMPT GENERATION *** ---
    try:
        # Use the keywords to generate a creative, natural-language prompt
        apiKey = "" # API key is automatically provided by the environment
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
        
        system_prompt = (   
            "You are a creative music prompt writer. You will be given a list of Python strings representing musical keywords (genres, instruments, tempo, vibe). "
            "Your job is to weave these keywords into a single, natural-language paragraph that is descriptive, evocative, and creative. This paragraph will be used as a prompt for a music generation AI. "
            "Do NOT just list the keywords. Weave them into a flowing description. Be creative and add some flair. The prompt should be about 2-3 sentences long. "
            "Do not use markdown. Do not add a preamble like 'Here is your prompt:'. Just return the new prompt text."
        )
        
        user_query = f"Here are the keywords to use: {json.dumps(final_phrases)}"

        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
        }
        
        headers = {'Content-Type': 'application/json'}
        
        # Make the synchronous API call
        response = requests.post(apiUrl, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status() # Will raise an error for bad responses
        
        result = response.json()
        final_prompt = result['candidates'][0]['content']['parts'][0]['text']
        
        # Clean up any potential markdown or extra spaces
        final_prompt = final_prompt.strip().replace("`", "")

    except Exception as e:
        # Fallback to the original comma-joining method if the API fails
        print(f"\nWarning: LLM prompt generation failed ({e}). Reverting to basic prompt.", file=sys.stderr)
        final_prompt = ", ".join(final_phrases)
        final_prompt = f"A track featuring: {final_prompt}."
    # --- *** END OF NEW SECTION *** ---

    return final_prompt

def generate_music(model, processor, device, prompt):
    """
    Generates a single piece of audio based on the provided prompt.
    Returns the filename of the saved audio.
    """
    try:
        # Calculate token limit for the desired audio length
        sample_rate = model.config.audio_encoder.sampling_rate
        frame_rate = model.config.audio_encoder.frame_rate
        max_new_tokens = int(AUDIO_LENGTH_S * frame_rate)

        # Prepare the inputs
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate the audio
        audio_values = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens
        )
        
        # --- Save the Audio ---
        timestamp = int(time.time())
        output_filename = f"music_gen_{timestamp}.wav"
        
        audio_numpy = audio_values[0].cpu().numpy()
        audio_mono = audio_numpy.squeeze()
        audio_int16 = (audio_mono * 32767).astype(np.int16)
        
        scipy.io.wavfile.write(output_filename, rate=sample_rate, data=audio_int16)
        
        return output_filename

    except Exception as e:
        print(f"An error occurred during generation: {e}", file=sys.stderr)
        return None

def generation_loop(model, processor, device):
    """
    Starts an interactive loop that takes user inputs and generates music.
    """
    profile_keys = list(USER_PROFILES.keys())
    
    while True:
        print("\n" + "-"*50, file=sys.stderr)
        
        # --- 1. Get User ID ---
        print(f"Available users: {', '.join(profile_keys)}", file=sys.stderr)
        user_id = input("Enter User ID (or 'exit' to quit): ").strip().lower()
        
        if user_id.lower() in ['exit', 'quit', 'q']:
            print("Exiting...", file=sys.stderr)
            break
        if user_id not in profile_keys:
            print(f"Invalid User ID. Please choose from: {profile_keys}", file=sys.stderr)
            continue
        
        profile = USER_PROFILES[user_id]
        print(f"Loaded profile: {profile['name']}", file=sys.stderr)

        # --- 2. Get Mood ---
        print(f"Available moods: {', '.join(VALID_MOODS)}", file=sys.stderr)
        mood = input("Enter detected mood: ").strip().lower()
        if mood not in VALID_MOODS:
            print(f"Invalid mood. Please choose from: {VALID_MOODS}", file=sys.stderr)
            continue

        # --- 3. Get Confidence ---
        confidence_str = input("Enter confidence score (0.0 to 1.0): ").strip()
        try:
            confidence = float(confidence_str)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError()
        except ValueError:
            print("Invalid confidence. Please enter a number between 0.0 and 1.0.", file=sys.stderr)
            continue
        
        # --- All inputs gathered, proceed to generation ---
        print("\nInputs received. Generating creative prompt via LLM...", file=sys.stderr)

        # 4. Use all inputs to create a single natural-language music prompt
        generated_prompt = create_music_prompt(
            mood,
            confidence,
            profile
        )
        
        # 5. Generate music audio using the MusicGen model
        print(f"LLM Prompt: {generated_prompt}", file=sys.stderr) # Print prompt to stderr for debugging
        output_file = generate_music(model, processor, device, generated_prompt)
        
        if output_file:
            # 6. Output *only* the prompt and the file path (to stdout)
            print(f"Prompt: {generated_prompt}")
            print(f"Music: {output_file}")

if __name__ == "__main__":
    # 1. Load the model ONCE
    model, processor, device = load_model()
    
    if model:
        # 2. Start the interactive controller loop
        generation_loop(model, processor, device)