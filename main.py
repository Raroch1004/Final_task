from core.graphics import DataVisualizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from core.test_files.test_fetcher import processed_messages

def main():
    # Загружаем модель и токенизатор
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")

    # Создаём объект визуализации
    visualizer = DataVisualizer(processed_messages, sentiment_model, sentiment_tokenizer)

    # Показываем график
    visualizer.create_general_timeline().show()

    # Сохраняем график
    visualizer.save_plot("sentiment_timeline.png")

if __name__ == "__main__":
    main()
