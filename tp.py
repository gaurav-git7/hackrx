import requests
import json

API_KEY = "AIzaSyBccHx_agsrHtWwATzTCibmmMQiVz-TX6U"  # 🔁 Replace with your actual API key
MODEL_NAME = "models/gemini-1.5-flash"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1/{MODEL_NAME}:generateContent?key={API_KEY}"

def check_gemini_key():
    headers = {
        "Content-Type": "application/json"
    }

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Hello! Can you confirm if my API key is working?"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 32,
            "topP": 1.0,
            "maxOutputTokens": 256
        }
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, data=json.dumps(body))
        data = response.json()

        if response.status_code == 200:
            print("✅ API key is valid and Gemini 1.5 Flash responded successfully.\n")
            answer = data['candidates'][0]['content']['parts'][0]['text']
            print("🤖 Gemini's response:")
            print(answer)

        else:
            error = data.get("error", {})
            print(f"❌ API request failed with status code {response.status_code}")
            print(f"Error Message: {error.get('message', 'No message')}")
            print(f"Error Status: {error.get('status', 'No status')}")

    except Exception as e:
        print("❌ An unexpected error occurred:", str(e))

if __name__ == "__main__":
    check_gemini_key()
