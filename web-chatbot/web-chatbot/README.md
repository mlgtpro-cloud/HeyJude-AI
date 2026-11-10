# AI Chatbot (Flask)

Web chatbot with streaming replies, markdown/code rendering, and chat memory within a session.
Works with any OpenAI-compatible free API (Groq, Gemini, OpenRouter...).

## Run
```
pip install -r requirements.txt
copy .env.example .env      (Mac/Linux: cp .env.example .env)
# open .env and paste your API key
python app.py
```
Open http://127.0.0.1:5000

## How it works
- `app.py`: Flask server. `/api/chat` forwards the conversation to the model API and streams tokens back.
- `templates/index.html`: chat UI. Keeps the message history in the browser and sends the last 20 messages each turn.
- Switch model or provider by changing `BASE_URL` and `MODEL` in `.env`.
