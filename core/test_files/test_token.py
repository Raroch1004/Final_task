import transformers
import torch
import pandas as pd

from core.test_files.test_fetcher import processed_messages

tokenizer = transformers.AutoTokenizer.from_pretrained('MonoHime/rubert-base-cased-sentiment-new')
model = transformers.AutoModelForSequenceClassification.from_pretrained('MonoHime/rubert-base-cased-sentiment-new')
sentiment_options = ["Neutral", "Positive", "Negative"]

for message in processed_messages:
    tokenized_input = tokenizer(message.content, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**tokenized_input)

    predicted_index = torch.argmax(model_output.logits).item()
    message.mood = sentiment_options[predicted_index]

results = {
    "Date": [],
    "Sentiment": [],
    "Content": [],
}

for message in processed_messages:
    results["Date"].append(message.timestamp)
    results["Sentiment"].append(message.mood)
    results["Content"].append(message.content)

output_df = pd.DataFrame(results)
output_df.to_csv("sentiment_results.csv", index=False)
print(processed_messages[0].mood)

