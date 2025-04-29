import os, requests
from dotenv import load_dotenv

load_dotenv()
token  = os.getenv("TELEGRAM_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")
url = f"https://api.telegram.org/bot{token}/sendMessage"

resp = requests.post(url, json={
    "chat_id": chat_id,
    "text": "Hello from my TradeSignalBot!",
})
resp.raise_for_status()
print("Message sent!")