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
    filters
)
import requests
from pymongo import MongoClient
from datetime import datetime

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

# ======================
# Configuration (add near top with other configs)
# ======================
REQUIRED_CHANNELS = {
    "BOT UPDATES": "https://t.me/Megahubbots",  # Channel username and invite link
    "PROMOTER CHANNEL": "https://t.me/Freenethubchannel"
}

# ======================
# Helper Functions
# ======================
async def check_membership(update: Update, user_id: int) -> bool:
    """Check if user is member of all required channels"""
    try:
        for channel in REQUIRED_CHANNELS.keys():
            chat_member = await update.get_bot().get_chat_member(
                chat_id=channel,
                user_id=user_id
            )
            if chat_member.status not in ['member', 'administrator', 'creator']:
                return False
        return True
    except Exception as e:
        logging.error(f"Error checking membership: {e}")
        return False

def get_join_keyboard():
    """Generate inline keyboard with join buttons"""
    keyboard = []
    for channel, link in REQUIRED_CHANNELS.items():
        keyboard.append([InlineKeyboardButton(f"Join {channel}", url=link)])
    keyboard.append([InlineKeyboardButton("✅ I've Joined", callback_data="check_join")])
    return InlineKeyboardMarkup(keyboard)

# ======================
# Modified Start Command
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Modified start command with membership check"""
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    
    # Check if user is member of required channels
    is_member = await check_membership(update, user_id)
    
    if not is_member:
        welcome_text = """
👋 *Welcome!* 

To use this bot, please join our official channels first:
"""
        for channel in REQUIRED_CHANNELS.keys():
            welcome_text += f"\n- {channel}"
            
        welcome_text += "\n\nAfter joining, click the button below to verify."
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=get_join_keyboard(),
            parse_mode="Markdown"
        )
        return
    
    # Existing start functionality
    if user_id not in user_sessions:
        bot_stats["total_users"] += 1
        await save_user(user_id, username)  # Save new user to DB
    
    await update.message.reply_text(
        "**Welcome to Multi-AI Bot!**\n\nChoose your preferred model:",
        reply_markup=markup,
        parse_mode="Markdown"
    )

# ======================
# Callback Handler for Join Verification
# ======================
async def verify_join(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle join verification callback"""
    query = update.callback_query
    user_id = query.from_user.id
    
    if query.data == "check_join":
        # Check membership again
        is_member = await check_membership(update, user_id)
        
        if is_member:
            await query.answer("✅ Verification successful! You can now use the bot.")
            
            # Edit message to remove buttons (keeping the image)
            await query.edit_message_caption(
                caption="**Verification Complete!**\n\nYou can now use all bot features.",
                reply_markup=None,
                parse_mode="Markdown"
            )
            
            # Show model selection keyboard in a new message
            await context.bot.send_message(
                chat_id=user_id,
                text="Please select a model:",
                reply_markup=markup
            )
            
            # Track user
            if user_id not in user_sessions:
                bot_stats["total_users"] += 1
                await save_user(user_id, query.from_user.username)
        else:
            await query.answer("❌ You haven't joined all channels yet!", show_alert=True)
            await query.edit_message_reply_markup(reply_markup=get_join_keyboard())

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
    
    # First update the user data (excluding message_count)
    users.update_one(
        {"user_id": user_id},
        {"$set": {
            "user_id": user_id,
            "username": username,
            "last_active": datetime.now(),
            "models_used": []  # Initialize if not exists
        }},
        upsert=True
    )
    
    # Then increment message_count separately
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
# Add this line for the welcome image (you'll replace this URL with your actual image URL)
WELCOME_IMAGE_URL = "https://envs.sh/keh.jpg"  # <-- REPLACE THIS WITH YOUR IMAGE URL

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Modified start command with membership check and welcome image"""
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    
    # Check if user is member of required channels
    is_member = await check_membership(update, user_id)
    
    if not is_member:
        welcome_text = """
👋 *Welcome to Our AI Bot!* 

📌 To access all features, please join our official channels first:
"""
        for channel in REQUIRED_CHANNELS.keys():
            welcome_text += f"\n- {channel}"
            
        welcome_text += "\n\nAfter joining, click the button below to verify."
        
        # Send image with caption
        await context.bot.send_photo(
            chat_id=user_id,
            photo=WELCOME_IMAGE_URL,  # <-- This is where your image URL will be used
            caption=welcome_text,
            reply_markup=get_join_keyboard(),
            parse_mode="Markdown"
        )
        return
    
    # Existing start functionality for verified users
    if user_id not in user_sessions:
        bot_stats["total_users"] += 1
        await save_user(user_id, username)  # Save new user to DB
    
    # Send welcome message with image for verified users
    await context.bot.send_photo(
        chat_id=user_id,
        photo=WELCOME_IMAGE_URL,  # <-- Same image or you can use a different one
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
# Error Handling
# ======================
async def handle_webhook(request):
    """Handle incoming Telegram updates"""
    try:
        if request.headers.get('X-Telegram-Bot-Api-Secret-Token') != 'YourSecretToken123':
            return web.Response(status=403)
        
        data = await request.json()
        update = Update.de_json(data, app.bot)
        await app.process_update(update)
        return web.Response(status=200)
    except Exception as e:
        logging.error(f"Error processing update: {e}")
        return web.Response(status=500)
    
# ======================
# indexes to improve performance
# ======================
def initialize_db():
    db = get_database()
    db.users.create_index("user_id", unique=True)
    db.users.create_index("last_active")

# ======================
# Health Check Endpoint
# ======================
async def health_check(request):
    try:
        db = get_database()
        # Simple ping to check DB connection
        db.command('ping')
        return web.Response(text="OK")
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return web.Response(status=503, text="Service Unavailable")

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

async def main():
    """Run the bot in webhook or polling mode"""
    global app
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register all handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("terms", terms))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("broadcast", broadcast))
    app.add_handler(CommandHandler("contactus", contactus))
    app.add_handler(MessageHandler(filters.Regex("^(🧠|🤖|💬|🦙)"), select_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # ... [other handlers] ...

    if os.environ.get('RENDER_EXTERNAL_HOSTNAME'):
        print("🌐 Running in webhook mode...")
        
        # Initialize the application
        await app.initialize()
        
        # Set up webhook properly
        await app.bot.set_webhook(
            url=WEBHOOK_URL,
            secret_token='YourSecretToken123',
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )
        
        # Configure aiohttp server with webhook endpoint
        server = web.Application()
        server.router.add_post(WEBHOOK_PATH, handle_webhook)
        
        # Add health check endpoint
        async def health_check(request):
            return web.Response(text="OK")
        
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
