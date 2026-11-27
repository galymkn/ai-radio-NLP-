"""Read public Telegram channels through a pre-authorized Telethon session."""
import asyncio
import logging
import os
import time

from . import RawArticle

log = logging.getLogger(__name__)


def fetch_channels(channels, cfg, is_known):
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    async def collect():
        client = TelegramClient(StringSession(os.environ['TELEGRAM_SESSION']),
                                int(os.environ['TELEGRAM_API_ID']), os.environ['TELEGRAM_API_HASH'],
                                timeout=20, request_retries=2, flood_sleep_threshold=0)
        result = []
        await client.connect()
        try:
            if not await client.is_user_authorized():
                raise RuntimeError('TELEGRAM_SESSION is not authorized; create it separately')
            cutoff = time.time() - cfg.max_age_hours * 3600
            for channel in channels:
                channel = channel.lstrip('@')
                try:
                    async for msg in client.iter_messages(channel, limit=cfg.max_items_per_feed):
                        if msg.date.timestamp() < cutoff:
                            break
                        url = f'https://t.me/{channel}/{msg.id}'
                        text = msg.raw_text or ''
                        if len(text) < cfg.min_text_chars or is_known(url):
                            continue
                        result.append(RawArticle(url, '@' + channel, 'telegram',
                                                 text.splitlines()[0], text, msg.date.timestamp()))
                except Exception:
                    log.exception('Telegram source %s failed', channel)
        finally:
            await client.disconnect()
        return result

    return asyncio.run(collect())
