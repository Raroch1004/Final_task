import os
import matplotlib.pyplot as plt
import pandas as pd
from core.test_files.test_fetcher import processed_messages
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DataVisualizer:
    def __init__(self, messages, sentiment_model, sentiment_tokenizer):
        self.messages = messages
        self.sentiment_model = sentiment_model
        self.sentiment_tokenizer = sentiment_tokenizer
        self.data_frame = self._prepare_data()

        if not os.path.exists("images"):
            os.makedirs("images")

    def analyze_sentiment(self, text):
        max_length = 512
        inputs = self.sentiment_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        outputs = self.sentiment_model(**inputs)
        return outputs.logits.argmax(dim=-1).item()

    def _prepare_data(self):
        data = {
            "Date": [msg.timestamp for msg in self.messages],
            "SentimentScore": [self.analyze_sentiment(msg.content) for msg in self.messages],
        }
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", dayfirst=True, errors="coerce")
        return df.groupby("Date").sum().reset_index()

    def create_general_timeline(self):
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.data_frame["Date"],
            self.data_frame["SentimentScore"],
            marker="o",
            linestyle="-",
            color="g",
            label="Sentiment Dynamics"
        )
        plt.title("Timeline of Sentiment Dynamics")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Sentiment Value")
        plt.xticks(rotation=45)
        plt.grid(visible=True)
        plt.legend()
        plt.tight_layout()
        return plt

    def save_plot(self, output_file: str):
        plt = self.create_general_timeline()
        plot_path = os.path.join("images", output_file)
        plt.savefig(plot_path)
        print(f"График сохранён как {plot_path}")

