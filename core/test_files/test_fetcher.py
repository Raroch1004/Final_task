from core.message_processor import DataFetcher

fetcher_instance = DataFetcher()
html_messages = fetcher_instance.extract_html()
processed_messages = fetcher_instance.parse_messages(html_messages)
