# BiLSTM Chatbot — Django + PyTorch

An intent-classification chatbot. A **Bidirectional LSTM** built in PyTorch
reads each message, predicts the intent behind it, and a Django view returns
a matching reply to a chat UI (plain HTML/CSS/JS, no extra frontend
framework required).

## Architecture

```
User message
   -> tokenize (regex word tokenizer)
   -> word indices (vocab built during training)
   -> nn.Embedding
   -> nn.LSTM(bidirectional=True)
   -> concat(final forward hidden, final backward hidden)
   -> Linear -> ReLU -> Dropout -> Linear
   -> softmax over intents
   -> if confidence >= threshold: sampled response for that intent
      else: fallback response
```

Model code: `chatbot/ml/model.py`
Training script: `chatbot/ml/train.py`
Training data: `chatbot/ml/intents.json`
Inference singleton used by Django: `chatbot/ml/inference.py`

## Project layout

```
chatbot_project/
├── manage.py
├── requirements.txt
├── chatbot_project/          # Django project (settings, urls)
└── chatbot/                  # Django app
    ├── views.py              # index page + /api/chat/ JSON endpoint
    ├── urls.py
    ├── ml/
    │   ├── model.py           # BiLSTM nn.Module
    │   ├── train.py           # trains and saves chatbot_model.pth
    │   ├── inference.py       # loads checkpoint, predicts intent
    │   ├── utils.py           # tokenizer + vocab helpers
    │   └── intents.json       # training patterns/responses
    ├── templates/chatbot/index.html
    └── static/chatbot/{css,js}
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the model

This has to be run once before the chatbot can answer anything — it
generates `chatbot/ml/chatbot_model.pth`, which is not included in this
zip because it depends on the exact PyTorch version you install.

```bash
python chatbot/ml/train.py
```

You'll see loss/accuracy printed every 25 epochs. On this small dataset it
trains in well under a minute on CPU.

## 2. Run Django

```bash
python manage.py migrate
python manage.py runserver
```

Visit **http://127.0.0.1:8000/** to chat.

## Extending it

- **Add intents**: edit `chatbot/ml/intents.json` (add a `tag`, some
  `patterns`, and candidate `responses`), then re-run `train.py`.
- **Bigger vocabulary / longer sentences**: raise `MAX_LEN` in `train.py`.
- **Stronger model**: raise `HIDDEN_DIM`/`EMBED_DIM`, add `NUM_LAYERS`, or
  swap the classifier head for a generative decoder (seq2seq) if you want
  free-form replies instead of intent-matched ones.
- **Confidence threshold**: tune `CHATBOT_CONFIDENCE_THRESHOLD` in
  `chatbot_project/settings.py` — lower it if real messages are being
  routed to the fallback intent too often.

## Streamlit alternative

If you'd rather skip Django entirely, the same `chatbot/ml` package
(`model.py`, `inference.py`, `utils.py`) is framework-agnostic — you can
drop it into a Streamlit app like this:

```python
import streamlit as st
from chatbot.ml.inference import ChatbotEngine

engine = ChatbotEngine.get_instance("chatbot/ml/chatbot_model.pth", "chatbot/ml/intents.json")

st.title("BiLSTM Chat")
message = st.chat_input("Type a message…")
if message:
    response, tag, confidence = engine.predict(message)
    st.chat_message("user").write(message)
    st.chat_message("assistant").write(response)
    st.caption(f"intent: {tag} · confidence: {confidence:.1%}")
```

Just say the word if you'd like this turned into a full standalone
Streamlit app instead.
