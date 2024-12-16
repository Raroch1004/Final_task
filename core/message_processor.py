import os
import torch
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class TelegramMessage:
    def __init__(self, content: str, timestamp):
        self.content = content
        self.timestamp = timestamp
        self.attributes = {}
        self.category = None
        self.mood = None

    def set_category(self, category: str):
        self.category = category

    def set_mood(self, mood: str):
        self.mood = mood

    def update_labels(self, category: str, mood: str):
        self.set_category(category)
        self.set_mood(mood)

    def prepare_attributes(self):
        self.attributes['timestamp'] = self.timestamp
        self.attributes['content'] = self.content


class DataFetcher:
    def __init__(self):
        self.base_path = os.path.dirname(__file__)
        self.html_file = "messages.html"

    def extract_html(self) -> list:
        with open(self.html_file, 'r', encoding='utf-8') as file:
            html_data = BeautifulSoup(file, 'html.parser')
            message_divs = html_data.find_all('div', class_='body')
            filtered_messages = [div for div in message_divs if div['class'] == ['body']]
            return filtered_messages

    def parse_messages(self, raw_messages: list) -> list:
        parsed_messages = []
        for raw_msg in raw_messages:
            if raw_msg.find('div', class_='text'):
                text = raw_msg.find('div', class_='text').get_text().strip()
                timestamp = raw_msg.find('div', class_='pull_right date details').get('title')

                timestamp = timestamp[:10].replace(".", "/")
                message_obj = TelegramMessage(text, timestamp)
                parsed_messages.append(message_obj)

        return parsed_messages


class TextAnalyzer:
    def __init__(self):
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        self.topic_model = None

    def initialize_sentiment_tools(self):
        if not self.sentiment_tokenizer or not self.sentiment_model:
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")

    def initialize_topic_tools(self):
        if not self.topic_model:
            self.topic_model = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    def analyze_sentiment(self, text: str) -> str:
        sentiment_labels = ["Neutral", "Positive", "Negative"]
        encoded_input = self.sentiment_tokenizer(text, padding=True, return_tensors="pt")

        with torch.no_grad():
            result = self.sentiment_model(**encoded_input)

        predicted_class = torch.argmax(result.logits).item()
        return sentiment_labels[predicted_class]

    def determine_topic(self, text: str) -> str:
        topics = ['Politics', 'Economy', 'Technology', 'World News', 'News']
        topic_result = self.topic_model(text, topics, multi_label=False)
        return topic_result['labels'][0]

    def reset_models(self):
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        self.topic_model = None

