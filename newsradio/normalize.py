"""Conservative speech preparation for English/Russian. Unknown forms stay unchanged."""
import re
from datetime import datetime
from num2words import num2words

ABBREVIATIONS = {
    'ru': {'США': 'Соединённые Штаты Америки', 'ООН': 'Организация Объединённых Наций',
           'РК': 'Республика Казахстан', 'км': 'километров', 'млн': 'миллионов', 'млрд': 'миллиардов'},
    'en': {'UN': 'United Nations', 'USA': 'United States', 'AI': 'artificial intelligence'},
}
MONTHS_RU = 'января февраля марта апреля мая июня июля августа сентября октября ноября декабря'.split()
MONTHS_EN = 'January February March April May June July August September October November December'.split()


def normalize_speech(text: str, language: str) -> str:
    if language not in ('en', 'ru'):
        return text
    def date(match):
        try:
            d = datetime.strptime(match[0], '%Y-%m-%d')
        except ValueError:
            return match[0]
        if language == 'ru':
            return f'{num2words(d.day, lang="ru", to="ordinal")} {MONTHS_RU[d.month-1]} {num2words(d.year, lang="ru")} года'
        return f'{MONTHS_EN[d.month-1]} {num2words(d.day, to="ordinal")}, {num2words(d.year, to="year")}'
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', date, text)
    for short, spoken in ABBREVIATIONS[language].items():
        text = re.sub(r'(?<!\w)' + re.escape(short) + r'(?!\w)', spoken, text)
    def number(match):
        value = match[0].replace(',', '.')
        return num2words(value, lang=language)
    # Decimal separator: dot in English, dot or comma in Russian. Thousands separators are left alone.
    pattern = r'(?<![\w.,])\d+(?:[.,]\d+)?(?![\w.,]\d)' if language == 'ru' else r'(?<![\w,])\d+(?:\.\d+)?(?![\w,])'
    text = re.sub(pattern, number, text)
    text = text.replace('%', ' процентов' if language == 'ru' else ' percent')
    return re.sub(r'\s+', ' ', text).strip()
