import os
import matplotlib.pyplot as plt
import pandas as pd
import torch

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
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True,
                                          max_length=max_length)
        outputs = self.sentiment_model(**inputs)
        return outputs.logits.argmax(dim=-1).item()

    def _convert_sentiment(self, sentiment_label):
        sentiment_map = {"Positive": 1, "Negative": -1, "Neutral": 0}
        return sentiment_map.get(sentiment_label, 0)

    def _prepare_data(self):
        data = {
            "Date": [msg.content for msg in self.messages],
            "SentimentScore": [
                self._convert_sentiment(self.analyze_sentiment(msg.content)) for msg in self.messages
            ]
        }
        df = pd.DataFrame(data)

        print(df["Date"])

        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", dayfirst=True, errors="coerce")
        return df.groupby("Date").sum().reset_index()

    def create_general_timeline(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data_frame["Date"], self.data_frame["SentimentScore"], marker="o", linestyle="-", color="g",
                 label="Sentiment Dynamics")
        plt.title("Timeline of Sentiment Dynamics")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Sentiment Value")
        plt.xticks(rotation=45)
        plt.grid(visible=True)
        plt.legend()
        plt.tight_layout()
        return plt

    def save_plot(self, output_file: str):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data_frame["Date"], self.data_frame["SentimentScore"], marker="o", linestyle="-", color="g",
                 label="Sentiment Dynamics")
        plt.title("Timeline of Sentiment Dynamics")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Sentiment Value")
        plt.xticks(rotation=45)
        plt.grid(visible=True)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join("images", output_file)
        plt.savefig(plot_path)
        print(f"График сохранён как {plot_path}")

    def create_histogram(self):
        sentiment_counts = self.data_frame["SentimentScore"].value_counts()

        plt.figure(figsize=(12, 6))
        plt.bar(sentiment_counts.index, sentiment_counts.values, color="skyblue", edgecolor="black")
        plt.title("Sentiment Score Histogram")
        plt.xlabel("Sentiment")
        plt.ylabel("Frequency")
        plt.xticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        return plt

    def create_topic_frequency_hist(self, topic_list: list, labels: pd.Series):
        topic_counts = labels.value_counts()
        filtered_counts = topic_counts.reindex(topic_list, fill_value=0)

        plt.figure(figsize=(10, 6))
        plt.bar(filtered_counts.index, filtered_counts.values, color="skyblue", edgecolor="black")
        plt.title("Topic Frequency")
        plt.xlabel("Topic")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        return plt

    def save_plt(self, output_file: str, plt: plt):
        plot_path = os.path.join("images", output_file)
        plt.savefig(plot_path)
        print(f"График сохранён как {plot_path}")


if __name__ == "__main__":
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")

    visualizer = DataVisualizer(processed_messages, sentiment_model, sentiment_tokenizer)

    visualizer.create_general_timeline().show()

    visualizer.save_plot("sentiment_timeline.png")

    visualizer.create_histogram().show()

    topic_list = ["Topic 1", "Topic 2"]
    labels = pd.Series(["Topic 1", "Topic 2", "Topic 1", "Topic 2"])
    visualizer.create_topic_frequency_hist(topic_list, labels).show()
