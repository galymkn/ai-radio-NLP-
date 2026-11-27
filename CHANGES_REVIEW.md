# Исправления по ревью

- Лимит попыток и backoff в отдельной таблице edit_attempts. Она автоматически добавляется при открытии старой SQLite/Postgres базы; существующие столбцы не меняются. failed остаётся до явного retry-failed; автоматически новых платных попыток не получает.
- Каждый запуск demo создаёт новый data/demo/run-… без удаления предыдущих данных. Рабочая база не используется. Повтор включён, cooldown=0.
- tts.normalize=false по умолчанию, включая Edge. Сам нормализатор не объявлен исправленным.
- В config.yaml видны telegram_backend, repeat_stories, max_attempts, retry_backoff_seconds, normalize. Рабочая ротация включена, но короткая очередь может ждать cooldown.
- Синхронные API-обработчики выполняются FastAPI в thread pool. /stream выносит БД и чтение файлов через asyncio.to_thread. Состояние слушателей защищено RLock.
- Первый старт хранится в broadcast_events: playing ACK для плеера, первый отправляемый фрагмент для stream. Это не счётчик фактически прослушанных клипов.
- export-labels выгружает тексты, URL, пустые метки, предсказанные кластеры, предложения и события. evaluate исключает пустую разметку, показывая её количество.

Команды:

```bash
python -m newsradio demo
python -m newsradio retry-failed
python -m newsradio export-labels --limit 50 --output data/annotations.json
python -m newsradio.evaluate data/annotations.json
```

Для обновления замените код, сохраните свой .env и data, добавьте новые параметры в свой config.yaml. Обновлять зависимости для этих исправлений не требуется. Данные проверки и границы — в VALIDATION.md.
