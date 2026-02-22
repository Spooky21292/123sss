import logging
import os
import time
from collections import deque
from typing import Deque, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Храним последние 10 пар user/assistant = до 20 сообщений на пользователя.
MAX_TURNS = 10
MAX_HISTORY_MESSAGES = MAX_TURNS * 2

user_histories: Dict[int, Deque[dict]] = {}
last_request_time: Dict[int, float] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот с ChatGPT.\n\n"
        "Отправь мне сообщение — я отвечу с учетом контекста диалога.\n"
        "Команды:\n"
        "/start — показать это сообщение\n"
        "/reset — сбросить историю диалога"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    last_request_time.pop(user_id, None)
    await update.message.reply_text("История очищена. Начнем заново 🙂")


def _is_rate_limited(user_id: int, min_interval_seconds: float = 2.0) -> bool:
    now = time.time()
    previous = last_request_time.get(user_id, 0)

    if now - previous < min_interval_seconds:
        return True

    last_request_time[user_id] = now
    return False


def _get_or_create_history(user_id: int) -> Deque[dict]:
    if user_id not in user_histories:
        user_histories[user_id] = deque(maxlen=MAX_HISTORY_MESSAGES)
    return user_histories[user_id]


def _build_input_messages(history: Deque[dict], user_text: str) -> List[dict]:
    messages = [{"role": "system", "content": "Ты полезный и дружелюбный ассистент."}]
    messages.extend(list(history))
    messages.append({"role": "user", "content": user_text})
    return messages


def _ask_openai(history: Deque[dict], user_text: str) -> str:
    response = openai_client.responses.create(
        model=MODEL,
        input=_build_input_messages(history, user_text),
    )
    return response.output_text.strip()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id
    user_text = (update.message.text or "").strip()

    if not user_text:
        await update.message.reply_text("Я пока умею работать только с текстом 🙂")
        return

    if _is_rate_limited(user_id):
        await update.message.reply_text("Подожди пару секунд 🙂")
        return

    history = _get_or_create_history(user_id)

    try:
        answer = _ask_openai(history, user_text)
        if not answer:
            answer = "Похоже, я не смог сформировать ответ. Попробуй переформулировать вопрос."

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})

        await update.message.reply_text(answer)

    except Exception as exc:
        logger.exception("OpenAI request failed for user_id=%s: %s", user_id, exc)
        await update.message.reply_text(
            "Сейчас есть проблема с сервисом ИИ. Попробуй еще раз чуть позже 🙏"
        )


def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started")
    application.run_polling()


if __name__ == "__main__":
    main()
