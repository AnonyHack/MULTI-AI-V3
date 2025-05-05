from asyncio.log import logger
import os
import asyncio
from dotenv import load_dotenv
from aiohttp import web
from telegram import InlineKeyboardButton, Update, ReplyKeyboardMarkup, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    CallbackQueryHandler
)
import requests
from pymongo import MongoClient
from datetime import datetime
from functools import wraps

import logging
logging.basicConfig(level=logging.DEBUG)

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

# Force join configuration
REQUIRED_CHANNELS = {
    "𝐌𝐀𝐈𝐍 𝐂𝐇𝐀𝐍𝐍𝐄𝐋": "https://t.me/Freenethubz",
    "𝐂𝐇𝐀𝐍𝐍𝐄𝐋 𝐀𝐍𝐍𝐎𝐔𝐍𝐂𝐄𝐌𝐄𝐍𝐓": "https://t.me/megahubbots"
}

# Welcome image URL
WELCOME_IMAGE_URL = "https://envs.sh/keh.jpg"

# ======================
# Helper Functions
# ======================
def is_admin(user_id: int) -> bool:
    """Check if user is admin"""
    return user_id == ADMIN_USER_ID

async def is_user_member(user_id, bot):
    """Check if user is member of all required channels"""
    for channel in REQUIRED_CHANNELS:
        try:
            chat_member = await bot.get_chat_member(chat_id=f"@{channel.split()[-1]}", user_id=user_id)
            if chat_member.status not in ["member", "administrator", "creator"]:
                return False
        except Exception as e:
            logger.error(f"Error checking channel membership: {e}")
            return False
    return True

async def check_membership(update: Update, user_id: int) -> bool:
    """Check if user is member of required channels"""
    if is_admin(user_id):
        return True
    return await is_user_member(user_id, update.get_bot())

def get_join_keyboard():
    """Create keyboard with join buttons"""
    buttons = []
    for channel_name, channel_url in REQUIRED_CHANNELS.items():
        buttons.append([InlineKeyboardButton(channel_name, url=channel_url)])
    buttons.append([InlineKeyboardButton("✅ Verify Membership", callback_data="check_join")])
    return InlineKeyboardMarkup(buttons)

async def ask_user_to_join(update: Update):
    """Show join buttons to user"""
    await update.message.reply_text(
        "🚨 To use this bot, you must join our channels first! 🚨\n\n"
        "Click the buttons below to join, then press '✅ Verify Membership' to verify.",
        reply_markup=get_join_keyboard(),
        parse_mode="Markdown"
    )

async def verify_membership(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle membership verification callback"""
    query = update.callback_query
    await query.answer()
    
    if await is_user_member(query.from_user.id, context.bot):
        await query.message.edit_text("✅ Verification successful! You can now use all bot commands.")
    else:
        await query.answer("❌ You haven't joined all channels yet!", show_alert=True)

def channel_required(func):
    """Decorator to enforce channel membership before executing any command"""
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        
        # Always allow admin commands
        if is_admin(user_id):
            return await func(update, context, *args, **kwargs)
            
        # Check channel membership
        if not await is_user_member(user_id, context.bot):
            await ask_user_to_join(update)
            return
        
        # If user is member, proceed with original command
        return await func(update, context, *args, **kwargs)
    return wrapped

# ======================
# Database Setup
# ======================
def get_database():
    """Initialize MongoDB connection with pooling"""
    client = MongoClient(
        os.getenv('MONGODB_URI'),
        maxPoolSize=50,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000
    )
    return client[os.getenv('MONGODB_DBNAME')]

async def save_user(user_id: int, username: str = None):
    """Save or update user in database"""
    db = get_database()
    users = db.users
    
    users.update_one(
        {"user_id": user_id},
        {"$set": {
            "user_id": user_id,
            "username": username,
            "last_active": datetime.now(),
            "models_used": []
        }},
        upsert=True
    )
    
    users.update_one(
        {"user_id": user_id},
        {"$inc": {"message_count": 1}}
    )

async def update_model_usage(user_id: int, model_name: str):
    """Update which models a user has used"""
    try:
        db = get_database()
        db.users.update_one(
            {"user_id": user_id},
            {
                "$addToSet": {"models_used": model_name},
                "$set": {"last_active": datetime.now()}
            }
        )
    except Exception as e:
        logging.error(f"Error updating model usage: {e}")

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
    """Start command with membership check and welcome image"""
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    
    # Check if user is member of required channels
    is_member = await check_membership(update, user_id)
    
    if not is_member:
        welcome_text = "👋 *Welcome to Our AI Bot!*\n\n📌 To access all features, please join our official channels first:"
        for channel_name in REQUIRED_CHANNELS:
            welcome_text += f"\n- {channel_name}"
        welcome_text += "\n\nAfter joining, click the button below to verify."
        
        await context.bot.send_photo(
            chat_id=user_id,
            photo=WELCOME_IMAGE_URL,
            caption=welcome_text,
            reply_markup=get_join_keyboard(),
            parse_mode="Markdown"
        )
        return
    
    # Existing start functionality for verified users
    if user_id not in user_sessions:
        bot_stats["total_users"] += 1
        await save_user(user_id, username)
    
    await context.bot.send_photo(
        chat_id=user_id,
        photo=WELCOME_IMAGE_URL,
        caption="**Welcome to Multi-AI Bot!**\n\nChoose your preferred model:",
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
    if not is_admin(update.message.from_user.id):
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
    if not is_admin(update.message.from_user.id):
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

async def contactus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Contact information command"""
    keyboard = [
        [InlineKeyboardButton("Contact Developer", url="https://t.me/AM_ITACHIUCHIHA")],
        [InlineKeyboardButton("Email Us", url="mailto:Freenethubbusiness@gmail.com")],
        [InlineKeyboardButton("Bot Updates", url="https://t.me/Megahubbots")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    contact_text = """
📞 𝐂𝐨𝐧𝐭𝐚𝐜𝐭 𝐔𝐬

For support or inquiries, feel free to reach out to us through the provided buttons.
We value your feedback and are here to assist you with any questions or issues you may have.

We'll respond within 24 hours!
"""
    await update.message.reply_text(contact_text, reply_markup=reply_markup)

# ======================
# Message Handlers
# ======================
async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle model selection"""
    model_choice = update.message.text
    user_id = update.message.from_user.id

    if model_choice in MODELS:
        user_sessions[user_id] = MODELS[model_choice]
        bot_stats["model_usage"][model_choice] = bot_stats["model_usage"].get(model_choice, 0) + 1
        await update_model_usage(user_id, model_choice)
        
        await update.message.reply_text(
            f"✅ You selected *{model_choice}*.\nSend me your message!",
            reply_markup=markup,
            parse_mode="Markdown"
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages with AI response"""
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    user_text = update.message.text

    # Check membership first
    is_member = await check_membership(update, user_id)
    if not is_member:
        await update.message.reply_text(
            "⚠️ Please join our channels first to use this bot.",
            reply_markup=get_join_keyboard()
        )
        return

    if user_id not in user_sessions:
        await update.message.reply_text("❗ Please select a model first using /start.")
        return

    await save_user(user_id, username)
    model_config = user_sessions[user_id]
    
    if model_config["api_type"] == "openrouter":
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
# Main Application
# ======================
async def handle_webhook(request):
    """Handle incoming Telegram updates"""
    if request.headers.get('X-Telegram-Bot-Api-Secret-Token') != 'YourSecretToken123':
        return web.Response(status=403)
    
    data = await request.json()
    update = Update.de_json(data, app.bot)
    await app.process_update(update)
    return web.Response(status=200)

async def health_check(request):
    """Health check endpoint"""
    try:
        db = get_database()
        db.command('ping')
        return web.Response(text="OK")
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return web.Response(status=503, text="Service Unavailable")

async def main():
    """Run the bot in webhook or polling mode"""
    global app
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("terms", terms))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("broadcast", broadcast))
    app.add_handler(CommandHandler("contactus", contactus))
    app.add_handler(MessageHandler(filters.Regex("^(🧠|🤖|💬|🦙)"), select_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(verify_membership))

    if os.environ.get('RENDER_EXTERNAL_HOSTNAME'):
        print("🌐 Running in webhook mode...")
        await app.initialize()
        await app.bot.set_webhook(
            url=WEBHOOK_URL,
            secret_token='YourSecretToken123',
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )
        
        server = web.Application()
        server.router.add_post(WEBHOOK_PATH, handle_webhook)
        server.router.add_get('/', health_check)
        
        runner = web.AppRunner(server)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', PORT)
        await site.start()
        
        print(f"🚀 Server running on port {PORT}")
        print(f"✅ Webhook ready at {WEBHOOK_URL}")
        
        while True:
            await asyncio.sleep(3600)
    else:
        print("🔄 Running in polling mode...")
        await app.run_polling()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Bot crashed: {e}")
        raise
