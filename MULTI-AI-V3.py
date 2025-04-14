import os
import asyncio
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
import requests
from pymongo import MongoClient
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ADMIN_USER_ID = int(os.getenv('ADMIN_USER_ID'))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PORT = int(os.environ.get("PORT", 10000))
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}"
WEBHOOK_URL = f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME')}{WEBHOOK_PATH}"

# User sessions tracking
user_sessions = {}
bot_stats = {
    "total_users": 0,
    "active_sessions": 0,
    "model_usage": {}
}

# ======================
# Database Setup
# ======================

def get_database():
    """Initialize MongoDB connection"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    return client[os.getenv('MONGODB_DBNAME')]

async def save_user(user_id: int, username: str = None):
    """Save or update user in database"""
    db = get_database()
    users = db.users
    
    user_data = {
        "user_id": user_id,
        "username": username,
        "last_active": datetime.now(),
        "models_used": [],
        "message_count": 0
    }
    
    users.update_one(
        {"user_id": user_id},
        {"$set": user_data, "$inc": {"message_count": 1}},
        upsert=True
    )

async def update_model_usage(user_id: int, model_name: str):
    """Update which models a user has used"""
    db = get_database()
    db.users.update_one(
        {"user_id": user_id},
        {"$addToSet": {"models_used": model_name}}
    )


# Model configuration
MODELS = {
    "🧠 DeepSeek R1": {
        "api_type": "openrouter",
        "model_name": "deepseek/deepseek-r1:free",
        "api_key": os.getenv('DEEPSEEK_R1_KEY')
    },
    "🦙 LLaMA V3": {
        "api_type": "openrouter",
        "model_name": "meta-llama/llama-4-scout:free",
        "api_key": os.getenv('LLAMA_KEY')
    },
    "💬 ChatGPT": {
        "api_type": "openrouter",
        "model_name": "openai/gpt-3.5-turbo",
        "api_key": os.getenv('CHATGPT_KEY')
    },
    "🤖 Mistral": {
        "api_type": "openrouter",
        "model_name": "mistralai/mistral-small-3.1-24b-instruct:free",
        "api_key": os.getenv('MISTRAL_KEY')
    }
}

# Keyboard layout
keyboard = [
    ["🧠 DeepSeek R1", "🤖 Mistral"],
    ["💬 ChatGPT", "🦙 LLaMA V3"]
]
markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)

# Terms and Conditions
TERMS_TEXT = """
📜 *Terms and Conditions*

1. This bot provides AI services for educational purposes only.
2. We don't store your conversation history permanently.
3. Don't share sensitive personal information.
4. The bot may have usage limits.
5. Models are provided by third-party APIs.

By using this bot, you agree to these terms.
"""

# ======================
# Command Handlers
# ======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Modified start command with user tracking"""
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    
    if user_id not in user_sessions:
        bot_stats["total_users"] += 1
        await save_user(user_id, username)  # Save new user to DB
    
    await update.message.reply_text(
        "**Welcome to Multi-AI Bot!**\n\nChoose your preferred model:",
        reply_markup=markup,
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help information"""
    help_text = """
🆘 *How to use this bot:*

1. Use /start to begin
2. Select an AI model from the menu
3. Just type to chat with the AI!
4. Change models anytime by selecting a new one

📝 *Available Commands:*
/help - Show this message
/terms - View terms and conditions
"""
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def terms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show terms and conditions"""
    await update.message.reply_text(TERMS_TEXT, parse_mode="Markdown")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command: show usage statistics"""
    if update.message.from_user.id != ADMIN_USER_ID:
        await update.message.reply_text("❌ Admin access required.")
        return
    
    stats_text = f"""
📊 *Bot Statistics*

• Total users: {bot_stats["total_users"]}
• Active sessions: {len(user_sessions)}
• Model usage:
"""
    for model, count in bot_stats["model_usage"].items():
        stats_text += f"  - {model}: {count}\n"
    
    await update.message.reply_text(stats_text, parse_mode="Markdown")

async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command: broadcast message to all users"""
    if update.message.from_user.id != ADMIN_USER_ID:
        await update.message.reply_text("❌ Admin access required.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /broadcast <message>")
        return
    
    message = " ".join(context.args)
    for user_id in user_sessions:
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=f"📢 *Announcement:*\n\n{message}",
                parse_mode="Markdown"
            )
        except Exception as e:
            print(f"Failed to send to {user_id}: {e}")
    
    await update.message.reply_text(f"✅ Broadcast sent to {len(user_sessions)} users.")

# ======================
# Message Handlers
# ======================

async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Modified model selection with DB tracking"""
    model_choice = update.message.text
    user_id = update.message.from_user.id

    if model_choice in MODELS:
        user_sessions[user_id] = MODELS[model_choice]
        
        # Update stats
        bot_stats["model_usage"][model_choice] = bot_stats["model_usage"].get(model_choice, 0) + 1
        await update_model_usage(user_id, model_choice)  # Save model usage to DB
        
        await update.message.reply_text(
            f"✅ You selected *{model_choice}*.\nSend me your message!",
            reply_markup=markup,
            parse_mode="Markdown"
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Modified message handler with DB tracking"""
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    user_text = update.message.text

    if user_id not in user_sessions:
        await update.message.reply_text("❗ Please select a model first using /start.")
        return

    # Update user activity in DB
    await save_user(user_id, username)

    model_config = user_sessions[user_id]
    
    if model_config["api_type"] == "openrouter":
        # OpenRouter API handling
        headers = {
            "Authorization": f"Bearer {model_config['api_key']}",
            "Content-Type": "application/json",
            "X-Title": "Multi-AI Chatbot"
        }

        payload = {
            "model": model_config["model_name"],
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_text}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }

        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
            
            # Better error handling
            if response.status_code == 401:
                await update.message.reply_text("🔐 Error: Invalid API key for this model")
                return
                
            data = response.json()

            if "choices" in data:
                reply = data["choices"][0]["message"]["content"]
                await update.message.reply_text(f"💡 *AI Response:*\n\n{reply}", parse_mode="Markdown")
            else:
                error_msg = data.get('error', {}).get('message', 'Unknown error')
                await update.message.reply_text(f"⚠️ Error: {error_msg}")
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to connect: {str(e)}")

# ======================
# Contact us
# ======================

async def contactus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Contact information command"""
    contact_text = """
📞 *Contact Us*

For support or inquiries:
- Email: Freenethubbusiness@gmail.com
- Telegram: @Freenethubz
- Admin Contact: @SILANDO @AM_ITACHIUCHIHA

We'll respond within 24 hours!
"""
    await update.message.reply_text(contact_text, parse_mode="Markdown")
    
    # Save the contact request
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    db = get_database()
    db.contact_requests.insert_one({
        "user_id": user_id,
        "username": username,
        "timestamp": datetime.now(),
        "message": "Used /contactus command"
    })


# ======================
# Main Application
# ======================

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("terms", terms))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("broadcast", broadcast))
    app.add_handler(CommandHandler("contactus", contactus))
    
    # Message handlers
    app.add_handler(MessageHandler(filters.Regex("^(🧠|🤖|💬|🦙)"), select_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Multi-AI bot is running...")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot is running on webhook!")

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    # Set webhook (delete old if needed)
    await app.bot.delete_webhook(drop_pending_updates=True)
    await app.bot.set_webhook(WEBHOOK_URL)

    print(f"🌐 Starting webhook at: {WEBHOOK_URL}")

    # Bind to port so Render knows we're listening
    await app.run_webhook(
    listen="0.0.0.0",
    port=PORT,
    path=WEBHOOK_PATH,
)

if __name__ == "__main__":
    asyncio.run(main())
