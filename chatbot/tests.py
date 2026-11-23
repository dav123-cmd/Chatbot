
import json
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse


class ChatbotAPITests(TestCase):

    def test_empty_message_returns_400(self):
        response = self.client.post(
            reverse("chatbot:chat_api"),
            data=json.dumps({"message": ""}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)

    def test_invalid_json_returns_400(self):
        response = self.client.post(
            reverse("chatbot:chat_api"),
            data="{invalid json}",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)

    @patch("chatbot.views.ChatbotEngine.get_instance")
    def test_valid_message_returns_chatbot_response(self, mock_engine):
        mock_engine.return_value.predict.return_value = (
            "Hello! How can I help you?",
            "greeting",
            0.95,
        )

        response = self.client.post(
            reverse("chatbot:chat_api"),
            data=json.dumps({"message": "Hello"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(data["intent"], "greeting")
        self.assertEqual(data["response"], "Hello! How can I help you?")
        self.assertEqual(data["confidence"], 0.95)
