import logging
import requests
import os
import time
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AlertSystem:
    def __init__(self, telegram_token=None, telegram_chat_id=None, cooldown_seconds=60):
        """
        Alert System for HEC using Telegram for free messages.
        """
        # Read from environment variables if not passed directly
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.cooldown_seconds = cooldown_seconds
        self.last_alerts = {}
        
    def send_alert(self, risk_level, details, location="Unknown Area"):
        """
        Trigger an alert if risk is High or Medium, with a cooldown per risk level to avoid spam.
        """
        current_time = time.time()
        if current_time - self.last_alerts.get(risk_level, 0) < self.cooldown_seconds:
            return False # Cooldown active, don't spam

        self.last_alerts[risk_level] = current_time

        if risk_level == "High":
            message = f"🚨 DANGER: High Risk Elephant Activity Detected!\n📍 Location: {location}\n📋 Details: {details}"
            self._send_telegram_alert(message)
            logging.warning(f"HIGH RISK ALERT SENT: {details}")
            
        elif risk_level == "Medium":
            message = f"⚠️ WARNING: Elephant Activity Near Village.\n📍 Location: {location}\n📋 Details: {details}"
            self._send_telegram_alert(message)
            logging.info(f"Medium Risk Alert Sent: {details}")
            
        else:
            logging.info(f"Low Risk Logged: {details}")

        return True

    def _send_telegram_alert(self, message):
        """
        Send a message via Telegram Bot API on a background thread.
        """
        if self.telegram_token and self.telegram_chat_id:
            import threading
            def _send():
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": message
                }
                try:
                    response = requests.post(url, json=payload, timeout=10)
                    if response.status_code == 200:
                        print(f"[TELEGRAM SENT] {message}")
                    else:
                        print(f"[TELEGRAM ERROR] Failed to send: {response.text}")
                except Exception as e:
                    print(f"[TELEGRAM EXCEPTION] {e}")
                    
            threading.Thread(target=_send, daemon=True).start()
        else:
            import threading
            def _mock_send():
                print(f"[MOCK TELEGRAM] Credentials missing. Message: {message}")
            threading.Thread(target=_mock_send, daemon=True).start()

if __name__ == "__main__":
    alert = AlertSystem()
    alert.send_alert("High", "Elephant charging detected", location="Zone A")
    alert.send_alert("High", "Spam test (should be ignored)", location="Zone A")
