Ecommerce Assistant Chatbot

A Django-based ecommerce assistant chatbot built with Python and PyTorch. The application provides a web interface where users can send messages and receive responses through an intent-classification model.

Features

- Web-based chatbot interface
- Django backend with a JSON chat endpoint
- PyTorch-based BiLSTM model for classifying message intents
- Confidence score returned with the predicted intent
- Fallback intent handling when model confidence is below the configured threshold
- Configured responses selected based on the predicted intent

How It Works

1. A user sends a message through the chatbot interface.
2. The frontend sends the message to the Django chat endpoint as a JSON POST request.
3. The chatbot inference component loads the trained model checkpoint and processes the message.
4. The BiLSTM model predicts an intent and produces a confidence score.
5. If confidence is below the configured threshold, the inference component uses a fallback intent.
6. The application selects a configured response and returns the result to the frontend as JSON.

The model is implemented in "chatbot/ml/model.py", while prediction and model-loading logic are handled in "chatbot/ml/inference.py".

Technology Stack

- Backend: Python, Django
- Machine Learning: PyTorch, NumPy
- Frontend: HTML, CSS, JavaScript
- Deployment: Render

Project Structure

- "chatbot/" — Django chatbot application
- "chatbot/ml/model.py" — BiLSTM chatbot model architecture
- "chatbot/ml/inference.py" — Model loading and intent prediction
- "chatbot_project/" — Django project settings and configuration
- "manage.py" — Django management entry point
- "requirements.txt" — Python dependencies

Live Demo

https://chatbot-lfq3.onrender.com

Getting Started

Prerequisites

- Python 3
- pip
- Git

1. Clone the repository

git clone https://github.com/dav123-cmd/Chatbot.git
cd Chatbot

2. Create and activate a virtual environment

Windows (Command Prompt):

python -m venv .venv
.venv\Scripts\activate

Windows (PowerShell):

python -m venv .venv
.venv\Scripts\Activate.ps1

macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Apply database migrations

python manage.py migrate

5. Run the development server

python manage.py runserver

Open http://127.0.0.1:8000/ in your browser.

Notes

- The application uses a trained model checkpoint for inference. The checkpoint and any required model configuration must be present in the locations expected by the inference code.
- This README describes the current chatbot implementation. It does not claim that the application performs live ecommerce transactions, payments, or order fulfilment.
- Do not commit secrets, private keys, or local environment files to the repository.

Future Improvements

- Add automated tests for the chat endpoint and inference behavior.
- Document the model training process and dataset.
- Add screenshots or a short demo GIF of the user interface.
- Improve API security and request validation before production use.

License

No license has been specified for this project.
