"""Global constants, defaults, and attribution."""

APP_NAME = "souplite"
CONFIG_FILE = "soup.yaml"
SOUP_DIR = ".soup"
EXPERIMENTS_DB = "experiments.db"

# Author & Inspiration
AUTHOR_NAME = "Muhtar Jaksilikov"
INSPIRATION_NOTE = "Inspired by Soup (Makazhan Alpamys). Re-engineered by Muhtar Jaksilikov for <= 2GB RAM limits."
GITHUB_URL = "https://github.com/MakazhanAlpamys/Soup"
ORIGINAL_GITHUB_URL = "https://github.com/MakazhanAlpamys/Soup"

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] + '\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + '\\n' }}"
    "{% endif %}{% endfor %}"
)
