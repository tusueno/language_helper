import streamlit as st
from openai import OpenAI
import os
import re
import json
import time
from datetime import datetime
import hashlib
from typing import Dict, List, Optional, Tuple
import logging
import tiktoken
import io
import tempfile
import wave

# Cache manager - prosty cache w pamięci
class SimpleCacheManager:
    """Prosty cache manager w pamięci"""
    def __init__(self, default_ttl=3600):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key):
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.default_ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (time.time(), value)
    
    def clear(self):
        self.cache.clear()

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Zmienne środowiskowe nie są już potrzebne - API key jest wprowadzany przez UI




# Funkcja do inicjalizacji sesji
def init_session_state():
    """Inicjalizacja stanu sesji"""
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = 0
    if 'total_cost' not in st.session_state:
        st.session_state.total_cost = 0.0
    if 'token_history' not in st.session_state:
        st.session_state.token_history = []
    if 'cost_history' not in st.session_state:
        st.session_state.cost_history = []
    if 'recorded_translation_text' not in st.session_state:
        st.session_state.recorded_translation_text = ""
    # Wersjonowanie kluczy widgetów audio, aby uniknąć ponownego przetwarzania po rerun
    if 'mic_widget_version' not in st.session_state:
        st.session_state.mic_widget_version = 0
    if 'file_widget_version' not in st.session_state:
        st.session_state.file_widget_version = 0
    # Zmienne związane z ćwiczeniem wymowy zostały usunięte dla kompatybilności ze Streamlit Cloud




# Funkcja do obliczania kosztów
def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Oblicz koszt na podstawie modelu i liczby tokenów"""
    # Ceny na 1000 tokenów (USD) - aktualizuj według najnowszych cen OpenAI
    pricing = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "tts-1": {"input": 0.015, "output": 0.0}  # TTS: $0.015 na 1000 znaków
    }
    
    if model == "tts-1":
        # TTS ma cenę na 1000 znaków, nie na tokeny
        total_chars = input_tokens * 4  # Przybliżenie: 1 token ≈ 4 znaki
        return (total_chars / 1000) * pricing["tts-1"]["input"]
    
    model_key = model if model in pricing else "gpt-4o"
    input_cost = (input_tokens / 1000) * pricing[model_key]["input"]
    output_cost = (output_tokens / 1000) * pricing[model_key]["output"]
    
    return input_cost + output_cost

# Funkcja do liczenia tokenów
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Policz liczbę tokenów w tekście"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback - przybliżone liczenie (1 token ≈ 4 znaki)
        return len(text) // 4

# Inicjalizacja OpenAI client
@st.cache_resource
def get_openai_client(api_key: str):
    """Inicjalizacja klienta OpenAI z cache'owaniem"""
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Błąd inicjalizacji OpenAI: {e}")
        return None

# Cache dla wyników API - używamy SimpleCacheManager
_cache_manager = SimpleCacheManager(default_ttl=3600)  # 1 godzina

def get_cached_response(cache_key: str):
    """Pobierz wynik z cache"""
    return _cache_manager.get(cache_key)

def set_cached_response(cache_key: str, data: any):
    """Zapisz wynik w cache"""
    _cache_manager.set(cache_key, data)  # 1 godzina

# Generowanie klucza cache
def generate_cache_key(text: str, function: str, **kwargs) -> str:
    """Generuj unikalny klucz cache"""
    content = f"{text}_{function}_{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(content.encode()).hexdigest()

# Funkcja do aktualizacji statystyk
def update_usage_stats(input_tokens: int, output_tokens: int, model: str):
    """Aktualizuj statystyki użycia"""
    total_tokens = input_tokens + output_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)
    
    st.session_state.total_tokens += total_tokens
    st.session_state.total_cost += cost
    
    # Dodaj do historii
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.token_history.append({
        "timestamp": timestamp,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "model": model
    })
    
    st.session_state.cost_history.append({
        "timestamp": timestamp,
        "cost": cost,
        "model": model
    })

# Funkcja do wyświetlania statystyk użycia
def display_usage_stats():
    """Wyświetl statystyki użycia API"""
    with st.sidebar:
        st.markdown("### 📊 Statystyki użycia API")
        
        # Aktualne statystyki
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔢 Łącznie tokenów", f"{st.session_state.total_tokens:,}")
        with col2:
            st.metric("💰 Łączny koszt", f"${st.session_state.total_cost:.4f}")
        
        # Szczegółowe statystyki
        if st.session_state.token_history:
            st.markdown("#### 📈 Ostatnie użycie:")
            latest = st.session_state.token_history[-1]
            # Oblicz koszt dla ostatniego użycia
            latest_cost = calculate_cost(latest['model'], latest['input_tokens'], latest['output_tokens'])
            st.info(f"""
            **Model:** {latest['model']}  
            **Tokeny wejściowe:** {latest['input_tokens']:,}  
            **Tokeny wyjściowe:** {latest['output_tokens']:,}  
            **Koszt:** ${latest_cost:.4f}
            """)
        
        # Historia kosztów
        if st.session_state.cost_history:
            with st.expander("📊 Historia kosztów"):
                for entry in reversed(st.session_state.cost_history[-10:]):  # Ostatnie 10
                    st.text(f"{entry['timestamp']}: ${entry['cost']:.4f} ({entry['model']})")
        
        # Reset statystyk
        if st.button("🔄 Resetuj statystyki", use_container_width=True):
            st.session_state.total_tokens = 0
            st.session_state.total_cost = 0.0
            st.session_state.token_history = []
            st.session_state.cost_history = []
            st.rerun()

# Funkcja do wprowadzania klucza API
def api_key_input():
    """Input dla klucza API użytkownika"""
    st.sidebar.markdown("### 🔑 Klucz API OpenAI")
    
    # Informacje o kluczu API
    st.sidebar.info("""
    **Aby używać aplikacji:**
    1. Zarejestruj się na [platform.openai.com](https://platform.openai.com)
    2. Wygeneruj klucz API w sekcji "API Keys"
    3. Wklej klucz poniżej
    4. **Twój klucz jest przechowywany lokalnie w sesji**
    """)
    
    # Input dla klucza API
    api_key = st.sidebar.text_input(
        "Wprowadź swój klucz API OpenAI:",
        type="password",
        placeholder="sk-...",
        value=st.session_state.api_key,
        help="Twój klucz API OpenAI (zaczyna się od 'sk-')"
    )
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        st.rerun()
    
    # Sprawdź format klucza
    if api_key and not api_key.startswith("sk-"):
        st.sidebar.error("❌ Nieprawidłowy format klucza API. Klucz powinien zaczynać się od 'sk-'")
        return None
    
    return api_key

# Słownik etykiet z lepszą organizacją
class Labels:
    """Zarządzanie etykietami w różnych językach"""
    
    @staticmethod
    def get_labels() -> Dict[str, Dict[str, str]]:
        return {
            "Tłumacz wielojęzyczny": {
                "Polski": "🌍 Tłumacz Wielojęzyczny",
                "English": "🌍 Multilingual Translator",
                "Deutsch": "🌍 Mehrsprachiger Übersetzer",
                "Українська": "🌍 Багатомовний перекладач",
                "Français": "🌍 Traducteur multilingue",
                "Español": "🌍 Traductor multilingüe",
                "العربية": "🌍 مترجم متعدد اللغات",
                "Arabski (libański dialekt)": "🌍 مترجم متعدد اللغات (لبناني)",
                "中文": "🌍 多语言翻译器",
                "日本語": "🌍 多言語翻訳者"
            },
            "Wpisz wiadomość do przetłumaczenia": {
                "Polski": "✍️ Wpisz wiadomość do przetłumaczenia",
                "English": "✍️ Enter message to translate",
                "Deutsch": "✍️ Geben Sie eine Nachricht zum Übersetzen ein",
                "Українська": "✍️ Введіть повідомлення dla переводу",
                "Français": "✍️ Entrez un message à traduire",
                "Español": "✍️ Introduce un mensaje para traducir",
                "العربية": "✍️ أدخل رسالة للترجمة",
                "Arabski (libański dialekt)": "✍️ أدخل رسالة للترجمة (لبناني)",
                "中文": "✍️ 输入要翻译的消息",
                "日本語": "✍️ 翻訳するメッセージを入力してください"
            },
            "Wprowadź tekst tutaj:": {
                "Polski": "📝 Wprowadź tekst tutaj:",
                "English": "📝 Enter text here:",
                "Deutsch": "📝 Text hier eingeben:",
                "Українська": "📝 Введіть текст тут:",
                "Français": "📝 Entrez le texte ici :",
                "Español": "📝 Introduce el texto aquí:",
                "العربية": "📝 أدخل النص هنا:",
                "Arabski (libański dialekt)": "📝 أدخل النص هنا (لبناني)",
                "中文": "📝 在此输入文本：",
                "日本語": "📝 ここにテキストを入力してください："
            },
            "Przetłumacz": {
                "Polski": "🚀 Przetłumacz",
                "English": "🚀 Translate",
                "Deutsch": "🚀 Übersetzen",
                "Українська": "🚀 Перекласти",
                "Français": "🚀 Traduire",
                "Español": "🚀 Traducir",
                "العربية": "🚀 ترجم",
                "Arabski (libański dialekt)": "🚀 ترجم (لبناني)",
                "中文": "🚀 翻译",
                "日本語": "🚀 翻訳する"
            },
            "Wyjaśnienia słów i gramatyki": {
                "Polski": "📚 Wyjaśnienia słów i gramatyki",
                "English": "📚 Word and Grammar Explanations",
                "Deutsch": "📚 Wort- und Grammatik-Erklärungen",
                "Українська": "📚 Пояснення слів і граматики",
                "Français": "📚 Explications des mots et de la grammaire",
                "Español": "📚 Explicaciones de palabras y gramática",
                "العربية": "📚 توضيح الكلمات والقواعد",
                "Arabski (libański dialekt)": "📚 توضيح الكلمات والقواعد (لبناني)",
                "中文": "📚 词语和语法解释",
                "日本語": "📚 単語と文法の説明"
            },
            "Wpisz zdanie lub tekst do wyjaśnienia:": {
                "Polski": "💭 Wpisz zdanie lub tekst do wyjaśnienia:",
                "English": "💭 Enter a sentence or text for explanation:",
                "Deutsch": "💭 Geben Sie einen Satz oder Text zur Erklärung ein:",
                "Українська": "💭 Введіть речення або текст для пояснення:",
                "Français": "💭 Entrez une phrase ou un texte à expliquer :",
                "Español": "💭 Introduce una frase o texto para explicación:",
                "العربية": "💭 أدخل جملة أو نصًا للتوضيح:",
                "Arabski (libański dialekt)": "💭 أدخل جملة أو نصًا للتوضيح (لبناني)",
                "中文": "💭 输入要解释的句子或文本：",
                "日本語": "💭 説明する文やテキストを入力してください："
            },
            "Wyjaśnij słowa i gramatykę": {
                "Polski": "🔍 Wyjaśnij słowa i gramatykę",
                "English": "🔍 Explain words and grammar",
                "Deutsch": "🔍 Wörter und Grammatik erklären",
                "Українська": "🔍 Пояснити слова і граматику",
                "Français": "🔍 Expliquer les mots et la grammaire",
                "Español": "🔍 Explicar palabras y gramática",
                "العربية": "🔍 اشرح الكلمات والقواعد",
                "Arabski (libański dialekt)": "🔍 اشرح الكلمات والقواعد (لبناني)",
                "中文": "🔍 解释单词和语法",
                "日本語": "🔍 単語と文法を説明する"
            },
            "Ładna wersja wypowiedzi – poprawa stylistyki": {
                "Polski": "✨ Ładna wersja wypowiedzi (bez tłumaczenia)",
                "English": "✨ Polished version – stylistic improvement",
                "Deutsch": "✨ Schöne Version – stilistische Verbesserung",
                "Українська": "✨ Гарна версія – покращення стилю",
                "Français": "✨ Version soignée – amélioration stylistique",
                "Español": "✨ Versión bonita – mejora estilística",
                "العربية": "✨ نسخة جميلة – تحسين الأسلوب",
                "Arabski (libański dialekt)": "✨ نسخة جميلة – تحسين الأسلوب (لبناني)",
                "中文": "✨ 优美版本 – 风格改进",
                "日本語": "✨ 美しいバージョン – スタイルの改善"
            },
            "Wpisz tekst do poprawy stylistycznej:": {
                "Polski": "🎨 Wpisz tekst do poprawy stylistycznej:",
                "English": "🎨 Enter text for stylistic improvement:",
                "Deutsch": "🎨 Geben Sie einen Text zur stilistischen Verbesserung ein:",
                "Українська": "🎨 Введіть текст для покращення стилю:",
                "Français": "🎨 Entrez un texte à améliorer stylistiquement :",
                "Español": "🎨 Introduce un texto para mejora estilística:",
                "العربية": "🎨 أدخل نصًا لتحسين الأسلوب:",
                "العربية (standardowa)": "🎨 أدخل نصًا لتحسين الأسلوب (فصحى)",
                "العربية (libański dialekt)": "🎨 أدخل نصًا لتحسين الأسلوب (لبناني)",
                "中文": "🎨 输入要改进风格的文本：",
                "日本語": "🎨 スタイル改善のためのテキストを入力してください："
            },
            "Popraw stylistykę i wygeneruj ładną wersję": {
                "Polski": "🎯 Popraw stylistykę i wygeneruj ładną wersję",
                "English": "🎯 Polish style and generate improved version",
                "Deutsch": "🎯 Stil verbessern und schöne Version erstellen",
                "Українська": "🎯 Покращити стиль і створити гарну версію",
                "Français": "🎯 Améliorer le style et générer une version soignée",
                "Español": "🎯 Mejorar el estilo y generar una versión bonita",
                "العربية": "🎯 حسّن الأسلوب وأنشئ نسخة جميلة",
                "العربية (standardowa)": "🎯 حسّن الأسلوب وأنشئ نسخة جميلة (فصحى)",
                "العربية (libański dialekt)": "🎯 حسّن الأسلوب وأنشئ نسخة جميلة (لبناني)",
                "中文": "🎯 改善风格并生成优美版本",
                "日本語": "🎯 スタイルを改善して美しいバージョンを生成する"
            },
            "Tłumaczenie z obcego języka + poprawa błędów": {
                "Polski": "🔧 Tłumaczenie z obcego języka + poprawa błędów",
                "English": "🔧 Translation from foreign language + error correction",
                "Deutsch": "🔧 Übersetzung aus Fremdsprache + Fehlerkorrektur",
                "Українська": "🔧 Переклад з іноземної мови + виправлення помилок",
                "Français": "🔧 Traduction d'une langue étrangère + correction des erreurs",
                "Español": "🔧 Traducción de idioma extranjero + corrección de errores",
                "العربية": "🔧 ترجمة من لغة أجنبية + تصحيح الأخطاء",
                "العربية (standardowa)": "🔧 ترجمة من لغة أجنبية + تصحيح الأخطاء (فصحى)",
                "العربية (libański dialekt)": "🔧 ترجمة من لغة أجنبية + تصحيح الأخطاء (لبناني)",
                "中文": "🔧 外语翻译+错误修正",
                "日本語": "🔧 外国語からの翻訳＋誤り修正"
            },
            "Wpisz tekst w języku obcym:": {
                "Polski": "🌐 Wpisz tekst w języku obcym:",
                "English": "🌐 Enter text in a foreign language:",
                "Deutsch": "🌐 Geben Sie einen Text in einer Fremdsprache ein:",
                "Українська": "🌐 Введіть текст іноземною мовою:",
                "Français": "🌐 Entrez un texte en langue étrangère :",
                "Español": "🌐 Introduce un texto en idioma extranjero:",
                "العربية": "🌐 أدخل نصًا بلغة أجنبية:",
                "العربية (standardowa)": "🌐 أدخل نصًا بلغة أجنبية (فصحى)",
                "العربية (libański dialekt)": "🌐 أدخل نصًا بلغة أجنبية (لبناني)",
                "中文": "🌐 输入外语文本：",
                "日本語": "🌐 外国語のテキストを入力してください："
            },
            "Wykryj język, popraw błędy i przetłumacz na polski": {
                "Polski": "🎯 Wykryj język, popraw błędy i przetłumacz na polski",
                "English": "🎯 Detect language, correct errors and translate to Polish",
                "Deutsch": "🎯 Sprache erkennen, Fehler korrigieren und ins Polnische übersetzen",
                "Українська": "🎯 Визначити мову, виправити помилки і перекласти на польську",
                "Français": "🎯 Détecter la langue, corriger les erreurs et traduire en polonais",
                "Español": "🎯 Detectar idioma, corregir errores y traducir al polaco",
                "العربية": "🎯 اكتشف اللغة وصحح الأخطاء وترجم إلى البولندية",
                "العربية (standardowa)": "🎯 اكتشف اللغة وصحح الأخطاء وترجم إلى البولندية (فصحى)",
                "العربية (libański dialekt)": "🎯 اكتشف اللغة وصحح الأخطاء وترجم إلى البولندية (لبناني)",
                "中文": "🎯 检测语言，纠正错误并翻译成波兰语",
                "日本語": "🎯 言語を検出し、誤りを修正してポーランド語に翻訳する"
            },
            "Fiszki ze słówek do nauki": {
                "Polski": "📖 Fiszki ze słówek do nauki",
                "English": "📖 Vocabulary flashcards for learning",
                "Deutsch": "📖 Vokabelkarten zum Lernen",
                "Українська": "📖 Картки слів для навчання",
                "Français": "📖 Fiches de vocabulaire pour apprendre",
                "Español": "📖 Tarjetas de vocabulario para aprender",
                "العربية": "📖 بطاقات المفردات للتعلم",
                "العربية (standardowa)": "📖 بطاقات المفردات للتعلم (فصحى)",
                "العربية (libański dialekt)": "📖 بطاقات المفردات للتعلم (لبناني)",
                "中文": "📖 学习词汇卡片",
                "日本語": "📖 学習用語彙カード"
            },
            "Wpisz tekst, z którego chcesz wygenerować fiszki:": {
                "Polski": "📝 Wpisz tekst, z którego chcesz wygenerować fiszki:",
                "English": "📝 Enter text to generate flashcards from:",
                "Deutsch": "📝 Geben Sie einen Text ein, aus dem Sie Vokabelkarten erstellen möchten:",
                "Українська": "📝 Введіть текст для створення карток:",
                "Français": "📝 Entrez un texte pour générer des fiches :",
                "Español": "📝 Introduce un texto para generar tarjetas:",
                "العربية": "📝 أدخل نصًا لإنشاء بطاقات المفردات:",
                "العربية (standardowa)": "📝 أدخل نصًا لإنشاء بطاقات المفردات (فصحى)",
                "العربية (libański dialekt)": "📝 أدخل نصًا لإنشاء بطاقات المفردات (لبناني)",
                "中文": "📝 输入要生成卡片的文本：",
                "日本語": "📝 カードを生成するためのテキストを入力してください："
            },
            "Wygeneruj fiszki": {
                "Polski": "🎯 Wygeneruj fiszki",
                "English": "🎯 Generate flashcards",
                "Deutsch": "🎯 Vokabelkarten erstellen",
                "Українська": "🎯 Створити картки",
                "Français": "🎯 Générer des fiches",
                "Español": "🎯 Generar tarjetas",
                "العربية": "🎯 إنشاء بطاقات المفردات",
                "العربية (standardowa)": "🎯 إنشاء بطاقات المفردات (فصحى)",
                "العربية (libański dialekt)": "🎯 إنشاء بطاقات المفردات (لبناني)",
                "中文": "🎯 生成卡片",
                "日本語": "🎯 カードを生成する"
            },
            "Pobierz fiszki jako plik .txt": {
                "Polski": "💾 Pobierz fiszki jako plik .txt",
                "English": "💾 Download flashcards as .txt file",
                "Deutsch": "💾 Vokabelkarten als .txt-Datei herunterladen",
                "Українська": "💾 Завантажити картки як файл .txt",
                "Français": "💾 Télécharger les fiches au format .txt",
                "Español": "💾 Descargar tarjetas como archivo .txt",
                "العربية": "💾 تحميل البطاقات كملف .txt",
                "العربية (standardowa)": "💾 تحميل البطاقات كملف .txt (فصحى)",
                "العربية (libański dialekt)": "💾 تحميل البطاقات كملف .txt (لبناني)",
                "中文": "💾 下载卡片为 .txt 文件",
                "日本語": "💾 .txt ファイルとしてカードをダウンロード"
            },
            # Etykiety dla funkcji audio
            # Etykieta audio została usunięta dla kompatybilności ze Streamlit Cloud
            "Lub nagraj swoją wypowiedź": {
                "Polski": "🎤 Lub nagraj swoją wypowiedź",
                "English": "🎤 Or record your speech",
                "Deutsch": "🎤 Oder nehmen Sie Ihre Rede auf",
                "Українська": "🎤 Або запишіть свою промову",
                "Français": "🎤 Ou enregistrez votre discours",
                "Español": "🎤 O graba tu discurso",
                "العربية": "🎤 أو سجل كلامك",
                "Arabski (libański dialekt)": "🎤 أو سجل كلامك (لبناني)",
                "中文": "🎤 或录制您的演讲",
                "日本語": "🎤 またはスピーチを録音する"
            },
            "Nagraj z mikrofonu": {
                "Polski": "🎤 Nagraj z mikrofonu",
                "English": "🎤 Record from microphone",
                "Deutsch": "🎤 Vom Mikrofon aufnehmen",
                "Українська": "🎤 Записати з мікрофона",
                "Français": "🎤 Enregistrer depuis le microphone",
                "Español": "🎤 Grabar desde el micrófono",
                "العربية": "🎤 سجل من الميكروفون",
                "Arabski (libański dialekt)": "🎤 سجل من الميكروفون (لبناني)",
                "中文": "🎤 从麦克风录制",
                "日本語": "🎤 マイクから録音する"
            },
            "Wczytaj plik audio": {
                "Polski": "📁 Wczytaj plik audio",
                "English": "📁 Load audio file",
                "Deutsch": "📁 Audiodatei laden",
                "Українська": "📁 Завантажити аудіофайл",
                "Français": "📁 Charger un fichier audio",
                "Español": "📁 Cargar archivo de audio",
                "العربية": "📁 تحميل ملف صوتي",
                "Arabski (libański dialekt)": "📁 تحميل ملف صوتي (لبناني)",
                "中文": "📁 加载音频文件",
                "日本語": "📁 音声ファイルを読み込む"
            },
            "Wyczyść tekst": {
                "Polski": "🗑️ Wyczyść tekst",
                "English": "🗑️ Clear text",
                "Deutsch": "🗑️ Text löschen",
                "Українська": "🗑️ Очистити текст",
                "Français": "🗑️ Effacer le texte",
                "Español": "🗑️ Limpiar texto",
                "العربية": "🗑️ مسح النص",
                "Arabski (libański dialekt)": "🗑️ مسح النص (لبناني)",
                "中文": "🗑️ 清除文本",
                "日本語": "🗑️ テキストをクリアする"
            },
            # Etykiety dla wyboru języka
            "Wybierz język docelowy": {
                "Polski": "🎯 Wybierz język docelowy",
                "English": "🎯 Select target language",
                "Deutsch": "🎯 Zielsprache auswählen",
                "Українська": "🎯 Виберіть цільову мову",
                "Français": "🎯 Sélectionner la langue cible",
                "Español": "🎯 Seleccionar idioma objetivo",
                "العربية": "🎯 اختر اللغة المستهدفة",
                "Arabski (libański dialekt)": "🎯 اختر اللغة المستهدفة (لبناني)",
                "中文": "🎯 选择目标语言",
                "日本語": "🎯 目標言語を選択"
            },
            "Losowy język": {
                "Polski": "🔄 Losowy język",
                "English": "🔄 Random language",
                "Deutsch": "🔄 Zufällige Sprache",
                "Українська": "🔄 Випадкова мова",
                "Français": "🔄 Langue aléatoire",
                "Español": "🔄 Idioma aleatorio",
                "العربية": "🔄 لغة عشوائية",
                "Arabski (libański dialekt)": "🔄 لغة عشوائية (لبناني)",
                "中文": "🔄 随机语言",
                "日本語": "🔄 ランダム言語"
            }
        }

# Lista języków do tłumaczenia
class Languages:
    """Zarządzanie językami tłumaczenia"""
    
    @staticmethod
    def get_languages() -> Dict[str, str]:
        return {
            "🇬🇧 Angielski": "English",
            "🇩🇪 Niemiecki": "German",
            "🇫🇷 Francuski": "French",
            "🇪🇸 Hiszpański": "Spanish",
            "🇮🇹 Włoski": "Italian",
            "🇺🇦 Ukraiński": "Ukrainian",
            "🇷🇺 Rosyjski": "Russian",
            "🇸🇦 Arabski": "Arabic",
            "🇨🇿 Czeski": "Czech",
            "🇸🇰 Słowacki": "Slovak",
            "🇵🇹 Portugalski": "Portuguese",
            "🇨🇳 Chiński": "Chinese",
            "🇯🇵 Japoński": "Japanese",
            "Arabski (libański dialekt)": "ar_lebanese"
        }

# Funkcje pomocnicze
class Utils:
    """Funkcje pomocnicze"""
    
    @staticmethod
    def validate_text(text: str) -> Tuple[bool, str]:
        """Walidacja tekstu wejściowego"""
        if not text or not text.strip():
            return False, "⚠️ Wprowadź tekst do przetworzenia."
        
        if len(text.strip()) > 5000:  # Maksymalna długość tekstu
            return False, f"⚠️ Tekst jest za długi. Maksymalna długość: 5000 znaków."
        
        return True, ""
    
    @staticmethod
    def create_success_message(message: str, icon: str = "✅") -> str:
        """Tworzenie komunikatu sukcesu"""
        return f"{icon} {message}"
    
    @staticmethod
    def create_error_message(message: str, icon: str = "❌") -> str:
        """Tworzenie komunikatu błędu"""
        return f"{icon} {message}"
    
    @staticmethod
    def create_info_message(message: str, icon: str = "ℹ️") -> str:
        """Tworzenie komunikatu informacyjnego"""
        return f"{icon} {message}"

# Klasa do obsługi API OpenAI
class OpenAIHandler:
    """Obsługa API OpenAI z lepszą obsługą błędów"""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.last_request_time = 0
    
    def _rate_limit_delay(self):
        """Opóźnienie między requestami"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < 1:  # Opóźnienie między requestami (1 sekunda)
            time.sleep(1 - time_since_last)
        self.last_request_time = time.time()
    
    def make_request(self, messages: List[Dict], model: str = "gpt-4o") -> Optional[str]:
        """Wykonanie requestu do OpenAI z obsługą błędów"""
        try:
            self._rate_limit_delay()
            
            # Policz tokeny wejściowe
            input_text = " ".join([msg["content"] for msg in messages])
            input_tokens = count_tokens(input_text, model)
            
            with st.spinner("🤔 Przetwarzam..."):
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
            
            # Policz tokeny wyjściowe i zaktualizuj statystyki
            output_text = response.choices[0].message.content
            output_tokens = count_tokens(output_text, model)
            
            # Aktualizuj statystyki użycia
            update_usage_stats(input_tokens, output_tokens, model)
            
            return output_text
            
        except Exception as e:
            error_msg = f"Błąd API OpenAI: {str(e)}"
            logger.error(error_msg)
            st.error(Utils.create_error_message(error_msg))
            return None
    
    def transcribe_audio(self, file_bytes: bytes, filename: str = "audio.wav") -> Optional[str]:
        """Transkrypcja audio w chmurze (OpenAI)"""
        try:
            self._rate_limit_delay()
            bio = io.BytesIO(file_bytes)
            bio.name = filename
            with st.spinner("🎤 Rozpoznaję mowę..."):
                resp = self.client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=bio
                )
            return getattr(resp, "text", None)
        except Exception as e:
            st.error(f"❌ Błąd transkrypcji: {e}")
            return None

# Klasa do zarządzania tłumaczeniami
class TranslationManager:
    """Zarządzanie tłumaczeniami"""
    
    def __init__(self, openai_handler: OpenAIHandler):
        self.openai_handler = openai_handler
        self.labels = Labels.get_labels()
        self.languages = Languages.get_languages()
    
    def translate_text(self, text: str, target_lang: str, lang: str, correct_errors: bool = False) -> Optional[Dict]:
        """Tłumaczenie tekstu z opcją poprawiania błędów"""
        # Sprawdź cache
        cache_key = generate_cache_key(text, "translate", target_lang=target_lang, correct_errors=correct_errors)
        cached_result = get_cached_response(cache_key)
        if cached_result:
            st.info("📋 Wynik z cache")
            return cached_result
        
        # Walidacja
        is_valid, error_msg = Utils.validate_text(text)
        if not is_valid:
            st.warning(error_msg)
            return None
        
        # Przygotuj prompt
        target_language = self.languages[target_lang]
        
        if correct_errors:
            prompt = (
                f"Wykryj język poniższego tekstu, popraw błędy gramatyczne i stylistyczne w oryginalnym języku, "
                f"a następnie przetłumacz poprawiony tekst na język {target_language}. "
                f"Jeśli język docelowy używa innego alfabetu niż łaciński, ZAWSZE dodaj transkrypcję (zapis fonetyczny w alfabecie łacińskim). "
                f"WAŻNE: Każda część wyniku musi być w osobnej linii!\n\n"
                f"Wyświetl wynik dokładnie w tym formacie:\n"
                f"Wykryty język: [nazwa języka]\n"
                f"Poprawiony tekst: [poprawiony tekst]\n"
                f"Tłumaczenie na {target_language}: [tłumaczenie]\n"
                f"Transkrypcja: [transkrypcja w alfabecie łacińskim - ZAWSZE dla języków z innym alfabetem]\n\n"
                f"Tekst: {text}"
            )
        else:
            prompt = (
                f"Przetłumacz poniższy tekst na język {target_language}. "
                f"Jeśli język docelowy używa innego alfabetu niż łaciński, włącz transkrypcję (zapis fonetyczny w alfabecie łacińskim) bezpośrednio w tłumaczeniu. "
                f"Wyświetl tylko tłumaczenie z wbudowaną transkrypcją.\n\nTekst: {text}"
            )
        
        # Wykonaj request
        if correct_errors:
            messages = [
                {"role": "system", "content": "Jesteś ekspertem językowym, który poprawia błędy i tłumaczy teksty. ZAWSZE formatuj wynik w osobnych liniach - każda część w nowej linii. Nie łącz wszystkiego w jeden ciąg tekstu. ZAWSZE dodawaj transkrypcję dla języków z innym alfabetem niż łaciński."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": "Jesteś tłumaczem i specjalistą od transkrypcji."},
                {"role": "user", "content": prompt}
            ]
        
        result = self.openai_handler.make_request(messages)
        if not result:
            return None
        
        # Parsuj wynik - transkrypcja jest już wbudowana w tłumaczenie
        translation_text = result.strip() if result else None
        
        # Jeśli poprawiamy błędy, wymuś formatowanie z nowymi liniami
        if correct_errors and translation_text:
            # Sprawdź czy wynik ma już nowe linie
            if '\n' not in translation_text:
                # Jeśli nie ma nowych linii, dodaj je automatycznie
                # Szukaj kluczowych fraz i dodaj nowe linie
                text_to_format = translation_text
                
                # Dodaj nowe linie przed kluczowymi frazami
                text_to_format = text_to_format.replace(" Poprawiony tekst:", "\nPoprawiony tekst:")
                text_to_format = text_to_format.replace(" Tłumaczenie na", "\nTłumaczenie na")
                text_to_format = text_to_format.replace(" Transkrypcja:", "\nTranskrypcja:")
                
                # Jeśli nadal nie ma nowych linii, spróbuj innego podejścia
                if '\n' not in text_to_format:
                    # Dodaj nowe linie po dwukropkach
                    text_to_format = text_to_format.replace(": ", ":\n")
                
                translation_text = text_to_format
        
        # Zapisz w cache
        result_data = {
            "translation": translation_text,
            "transcription": None,  # Transkrypcja jest wbudowana w tłumaczenie
            "original_text": text,
            "target_language": target_lang,
            "timestamp": datetime.now().isoformat()
        }
        set_cached_response(cache_key, result_data)
        
        return result_data
    
    def _extract_section(self, text: str, section_start: str) -> str:
        """Wyciąga określoną sekcję z tekstu wynikowego"""
        try:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(section_start):
                    # Wyciągnij tekst po dwukropku
                    colon_pos = line.find(':')
                    if colon_pos > 0:
                        content = line[colon_pos + 1:].strip()
                        # Jeśli następna linia nie zaczyna się od nowej sekcji, dodaj ją do treści
                        if i + 1 < len(lines) and not any(lines[i + 1].strip().startswith(x) for x in ["Wykryty język:", "Poprawiony tekst:", "Tłumaczenie na", "Transkrypcja:"]):
                            content += " " + lines[i + 1].strip()
                        return content if content else "Brak danych"
            return "Brak danych"
        except:
            return "Błąd odczytu"
    
    def generate_audio(self, text: str) -> Optional[bytes]:
        """Generowanie audio z tekstu (tylko tłumaczenie, bez dodatkowych informacji)"""
        try:
            # Wyciągnij tylko tłumaczenie z tekstu
            audio_text = text
            
            # Jeśli tekst zawiera "Tłumaczenie na" - wyciągnij tylko tę część
            if "Tłumaczenie na" in text:
                lines = text.split('\n')
                for line in lines:
                    if line.strip().startswith("Tłumaczenie na"):
                        # Wyciągnij tekst po dwukropku
                        colon_pos = line.find(':')
                        if colon_pos > 0:
                            audio_text = line[colon_pos + 1:].strip()
                            break
            # Jeśli tekst zawiera "Transkrypcja:" - wyciągnij tylko transkrypcję
            elif "Transkrypcja:" in text:
                lines = text.split('\n')
                for line in lines:
                    if line.strip().startswith("Transkrypcja:"):
                        # Wyciągnij tekst po dwukropku
                        colon_pos = line.find(':')
                        if colon_pos > 0:
                            audio_text = line[colon_pos + 1:].strip()
                            break
            # Jeśli tekst zawiera nawiasy - usuń wszystko po pierwszym nawiasie (stara logika)
            elif '(' in text:
                first_open = text.find('(')
                if first_open > 0:
                    audio_text = text[:first_open].strip()
            
            # Policz tokeny dla TTS (TTS używa innego systemu liczenia)
            input_tokens = count_tokens(audio_text, "gpt-4o")
            
            tts_response = self.openai_handler.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=audio_text
            )
            
            # TTS ma stałą cenę na 1000 znaków, nie na tokeny
            # Przyjmujemy 1 token ≈ 4 znaki
            output_tokens = len(audio_text) // 4
            
            # Aktualizuj statystyki użycia (TTS ma inne ceny)
            update_usage_stats(input_tokens, output_tokens, "tts-1")
            
            return tts_response.content
        except Exception as e:
            st.error(Utils.create_error_message(f"Błąd generowania audio: {e}"))
            return None

# Klasa do zarządzania wyjaśnieniami
class ExplanationManager:
    """Zarządzanie wyjaśnieniami słów i gramatyki"""
    
    def __init__(self, openai_handler: OpenAIHandler):
        self.openai_handler = openai_handler
    
    def explain_text(self, text: str) -> Optional[str]:
        """Wyjaśnienie tekstu"""
        # Sprawdź cache
        cache_key = generate_cache_key(text, "explain")
        cached_result = get_cached_response(cache_key)
        if cached_result:
            st.info("📋 Wynik z cache")
            return cached_result
        
        # Walidacja
        is_valid, error_msg = Utils.validate_text(text)
        if not is_valid:
            st.warning(error_msg)
            return None
        
        # Przygotuj prompt
        prompt = (
            "Wyjaśnij trudniejsze słowa i konstrukcje gramatyczne w poniższym tekście. "
            "Podaj krótkie definicje słówek oraz opisz użyte struktury gramatyczne w prosty sposób.\n\n"
            f"Tekst: {text}"
        )
        
        # Wykonaj request
        messages = [
            {"role": "system", "content": "Jesteś nauczycielem języka obcego, który tłumaczy słowa i gramatykę prostym językiem."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.openai_handler.make_request(messages)
        if result:
            set_cached_response(cache_key, result)
        
        return result

# Klasa do zarządzania stylistyką
class StyleManager:
    """Zarządzanie poprawą stylistyki"""
    
    def __init__(self, openai_handler: OpenAIHandler):
        self.openai_handler = openai_handler
    
    def improve_style(self, text: str) -> Optional[str]:
        """Poprawa stylistyki tekstu"""
        # Sprawdź cache
        cache_key = generate_cache_key(text, "style")
        cached_result = get_cached_response(cache_key)
        if cached_result:
            st.info("📋 Wynik z cache")
            return cached_result
        
        # Walidacja
        is_valid, error_msg = Utils.validate_text(text)
        if not is_valid:
            st.warning(error_msg)
            return None
        
        # Przygotuj prompt
        prompt = (
            "Popraw stylistykę, gramatykę i płynność poniższej wypowiedzi. "
            "WAŻNE: Zwróć tekst w tym samym języku, w którym został napisany. "
            "NIE tłumacz na polski, tylko popraw stylistykę i gramatykę w oryginalnym języku. "
            "Zwróć tekst w ładnej, naturalnej wersji, odpowiedniej dla native speakera.\n\n"
            f"Tekst: {text}"
        )
        
        # Wykonaj request
        messages = [
            {"role": "system", "content": "Jesteś ekspertem językowym i stylistą. Twoim zadaniem jest poprawianie stylistyki i gramatyki w tym samym języku, w którym został napisany tekst. NIE tłumacz tekstu na inne języki."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.openai_handler.make_request(messages)
        if result:
            set_cached_response(cache_key, result)
        
        return result

# Klasa do zarządzania tłumaczeniami z poprawą błędów
class CorrectionManager:
    """Zarządzanie tłumaczeniami z poprawą błędów"""
    
    def __init__(self, openai_handler: OpenAIHandler):
        self.openai_handler = openai_handler
    
    def correct_and_translate(self, text: str) -> Optional[str]:
        """Poprawa błędów i tłumaczenie"""
        # Sprawdź cache
        cache_key = generate_cache_key(text, "correct")
        cached_result = get_cached_response(cache_key)
        if cached_result:
            st.info("📋 Wynik z cache")
            return cached_result
        
        # Walidacja
        is_valid, error_msg = Utils.validate_text(text)
        if not is_valid:
            st.warning(error_msg)
            return None
        
        # Przygotuj prompt
        prompt = (
            "Wykryj język poniższego tekstu, popraw błędy gramatyczne i stylistyczne, "
            "a następnie przetłumacz go na polski. "
            "Wyświetl wynik w formacie:\n"
            "Wykryty język: ...\n"
            "Poprawiony tekst: ...\n"
            "Tłumaczenie na polski: ...\n\n"
            f"Tekst: {text}"
        )
        
        # Wykonaj request
        messages = [
            {"role": "system", "content": "Jesteś ekspertem językowym i tłumaczem."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.openai_handler.make_request(messages)
        if result:
            set_cached_response(cache_key, result)
        
        return result

# Klasa do zarządzania fiszkami
class FlashcardManager:
    """Zarządzanie fiszkami z generowaniem PDF"""
    
    def __init__(self, openai_handler: OpenAIHandler):
        self.openai_handler = openai_handler
    
    def generate_flashcards(self, text: str) -> Optional[Dict]:
        """Generowanie fiszek z tekstu i zwracanie struktury danych"""
        # Sprawdź cache
        cache_key = generate_cache_key(text, "flashcards")
        cached_result = get_cached_response(cache_key)
        if cached_result:
            st.info("📋 Wynik z cache")
            return cached_result
        
        # Walidacja
        is_valid, error_msg = Utils.validate_text(text)
        if not is_valid:
            st.warning(error_msg)
            return None
        
        # Przygotuj prompt
        prompt = (
            "Wypisz listę najważniejszych i najciekawszych słówek z poniższego tekstu. "
            "Do każdego słowa podaj krótką definicję po polsku oraz przykład użycia w zdaniu. "
            "WAŻNE: Odpowiedz TYLKO w formacie JSON, bez żadnych dodatkowych komentarzy, markdown lub tekstu przed lub po JSON.\n"
            "Format:\n"
            "{\n"
            '  "flashcards": [\n'
            '    {\n'
            '      "word": "słówko",\n'
            '      "definition": "definicja po polsku",\n'
            '      "example": "przykład użycia w zdaniu"\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            f"Tekst: {text}"
        )
        
        # Wykonaj request
        messages = [
            {"role": "system", "content": "Jesteś nauczycielem języka obcego, który tworzy fiszki do nauki słówek. ZAWSZE odpowiadaj TYLKO w formacie JSON, bez żadnych dodatkowych komentarzy, markdown, ani tekstu przed lub po JSON. Twoja odpowiedź musi zaczynać się od { i kończyć na }."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.openai_handler.make_request(messages)
        if not result:
            st.error("❌ Nie otrzymano odpowiedzi od OpenAI")
            return None
            

        
        try:
            # Próbuj sparsować JSON
            import json
            parsed_result = json.loads(result)
            
            # Sprawdź czy struktura jest poprawna
            if isinstance(parsed_result, dict) and "flashcards" in parsed_result:
                if isinstance(parsed_result["flashcards"], list) and len(parsed_result["flashcards"]) > 0:
                    set_cached_response(cache_key, parsed_result)
                    return parsed_result
            
            # Jeśli struktura jest niepoprawna, spróbuj naprawić
            st.warning("⚠️ Struktura odpowiedzi jest niepoprawna. Próbuję naprawić...")
            return {"flashcards": [{"word": "Błąd struktury", "definition": "Odpowiedź ma niepoprawną strukturę", "example": "Spróbuj ponownie"}]}
            
        except json.JSONDecodeError as e:
            # Jeśli nie udało się sparsować JSON, spróbuj naprawić
            st.warning(f"⚠️ Błąd parsowania JSON: {e}")
            st.info("🔄 Próbuję naprawić odpowiedź...")
            
            # Spróbuj wyciągnąć słówka z tekstu
            try:
                # Usuń markdown i inne formatowanie
                cleaned_result = result.replace("```json", "").replace("```", "").strip()
                if cleaned_result:
                    parsed_result = json.loads(cleaned_result)
                    if isinstance(parsed_result, dict) and "flashcards" in parsed_result:
                        if isinstance(parsed_result["flashcards"], list) and len(parsed_result["flashcards"]) > 0:
                            set_cached_response(cache_key, parsed_result)
                            return parsed_result
            except:
                pass
            
            # Jeśli wszystko się nie udało, zwróć błąd
            st.error("❌ Nie udało się naprawić odpowiedzi")
            return {"flashcards": [{"word": "Błąd parsowania", "definition": f"Nie udało się sparsować: {result[:100]}...", "example": "Spróbuj ponownie"}]}
        
        return None
    
    def generate_images(self, flashcards_data: Dict, size_choice: str = "Duże (800×600)", format_choice: str = "PNG (najlepsza jakość)", quality_choice: str = "Wysoka") -> Optional[bytes]:
        """Generuje obrazy PNG z fiszkami w wybranym rozmiarze"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
        except ImportError:
            st.error("❌ Brak biblioteki Pillow. Zainstaluj: pip install Pillow")
            return None
        
        try:
            # Przygotowanie danych
            flashcards = flashcards_data.get("flashcards", [])
            if not flashcards:
                st.error("❌ Brak danych fiszek do wygenerowania obrazów")
                return None
            
            # Ustawienia obrazu - wybór rozmiaru
            if "Duże" in size_choice:
                card_width, card_height = 800, 600
                margin = 50
                font_large_size, font_medium_size, font_small_size = 32, 24, 18
            elif "Średnie" in size_choice:
                card_width, card_height = 600, 450
                margin = 40
                font_large_size, font_medium_size, font_small_size = 24, 18, 14
            else:  # Małe
                card_width, card_height = 400, 300
                margin = 30
                font_large_size, font_medium_size, font_small_size = 18, 14, 10
            
            cards_per_row = 2
            cards_per_col = 2
            
            # Rozmiar całego obrazu
            total_width = cards_per_row * card_width + (cards_per_row + 1) * margin
            total_height = cards_per_col * card_height + (cards_per_col + 1) * margin + 100  # +100 na tytuł
            
            # Tworzenie obrazu
            img = Image.new('RGB', (total_width, total_height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Próba załadowania czcionki z obsługą polskich znaków
            try:
                # Próbujemy różne czcionki
                font_large = ImageFont.truetype("arial.ttf", font_large_size)  # Windows Arial
            except:
                try:
                    font_large = ImageFont.truetype("DejaVuSans.ttf", font_large_size)  # Linux
                except:
                    font_large = ImageFont.load_default()  # Domyślna czcionka
            
            try:
                font_medium = ImageFont.truetype("arial.ttf", font_medium_size)
            except:
                font_medium = ImageFont.load_default()
            
            try:
                font_small = ImageFont.truetype("arial.ttf", font_small_size)
            except:
                font_small = ImageFont.load_default()
            
            # Tytuł
            title = "📚 FISZKI DO NAUKI"
            title_bbox = draw.textbbox((0, 0), title, font=font_large)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (total_width - title_width) // 2
            draw.text((title_x, 20), title, fill='#1f77b4', font=font_large)
            
            # Generowanie fiszek
            for i, card in enumerate(flashcards):
                if i >= cards_per_row * cards_per_col:
                    break
                
                # Pozycja fiszki
                row = i // cards_per_row
                col = i % cards_per_row
                
                x = margin + col * (card_width + margin)
                y = 100 + row * (card_height + margin)
                
                # Rysowanie ramki fiszki
                draw.rectangle([x, y, x + card_width, y + card_height], 
                             outline='#1f77b4', width=3, fill='#f8f9fa')
                
                # Linia podziału
                draw.line([x, y + card_height//2, x + card_width, y + card_height//2], 
                         fill='#ff7f0e', width=2)
                
                # Słówko
                word = card.get("word", "")[:30]
                word_bbox = draw.textbbox((0, 0), word, font=font_large)
                word_width = word_bbox[2] - word_bbox[0]
                word_x = x + (card_width - word_width) // 2
                draw.text((word_x, y + 20), "SŁÓWKO:", fill='#1f77b4', font=font_medium)
                draw.text((word_x, y + 60), word, fill='#333', font=font_large)
                
                # Definicja
                definition = card.get("definition", "")[:60]
                definition_bbox = draw.textbbox((0, 0), definition, font=font_small)
                definition_width = definition_bbox[2] - definition_bbox[0]
                definition_x = x + (card_width - definition_width) // 2
                draw.text((definition_x, y + card_height//2 + 20), "DEFINICJA:", fill='#1f77b4', font=font_medium)
                draw.text((definition_x, y + card_height//2 + 60), definition, fill='#555', font=font_small)
                
                # Przykład
                example = card.get("example", "")[:80]
                example_bbox = draw.textbbox((0, 0), example, font=font_small)
                example_width = example_bbox[2] - example_bbox[0]
                example_x = x + (card_width - example_width) // 2
                draw.text((example_x, y + card_height - 80), "PRZYKŁAD:", fill='#1f77b4', font=font_medium)
                draw.text((example_x, y + card_height - 50), example, fill='#666', font=font_small)
            
            # Konwersja do bytes - wybór formatu i jakości
            buffer = io.BytesIO()
            
            if "JPG" in format_choice:
                # JPG z wyborem jakości
                if "Wysoka" in quality_choice:
                    quality = 95
                elif "Średnia" in quality_choice:
                    quality = 80
                else:  # Niska
                    quality = 60
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
            else:
                # PNG - zawsze wysoka jakość
                img.save(buffer, format='PNG', optimize=True)
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            st.error(f"❌ Błąd generowania obrazów: {str(e)}")
            return None

# Klasa do rozpoznawania mowy
class SpeechRecognitionManager:
    """Zarządzanie rozpoznawaniem mowy"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def get_audio_from_microphone(self) -> Optional[str]:
        """Nagrywanie audio z mikrofonu i konwersja na tekst"""
        try:
            # Użyj domyślnego mikrofonu
            with sr.Microphone() as source:
                # Dostosuj do hałasu otoczenia
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
                # Ustaw parametry dla lepszego nagrywania
                self.recognizer.energy_threshold = 200
                self.recognizer.dynamic_energy_threshold = False
                self.recognizer.pause_threshold = 2.0
                self.recognizer.non_speaking_duration = 2.0
                
                # Nagrywaj audio
                audio = self.recognizer.listen(source, timeout=120, phrase_time_limit=120)
                
            # Konwertuj audio na tekst
            text = self.recognizer.recognize_google(audio, language='pl-PL')
            
            if text:
                return text
            else:
                return None
                    
        except sr.WaitTimeoutError:
            raise Exception("Przekroczono limit czasu oczekiwania na mowę. Spróbuj ponownie.")
        except sr.UnknownValueError:
            raise Exception("Nie udało się rozpoznać mowy. Mów wyraźniej i w normalnym tempie.")
        except sr.RequestError as e:
            raise Exception(f"Błąd serwisu rozpoznawania mowy: {e}. Sprawdź połączenie internetowe.")
        except Exception as e:
            raise Exception(f"Błąd nagrywania: {e}. Sprawdź czy mikrofon działa.")
    
    def get_audio_from_file(self, audio_file) -> Optional[str]:
        """Konwersja audio z pliku na tekst"""
        try:
            # Wczytaj plik audio
            audio = sr.AudioFile(audio_file)
            
            with audio as source:
                # Konwertuj audio na tekst
                text = self.recognizer.recognize_google(audio, language='pl-PL')
                
            if text:
                return text
            else:
                return None
                    
        except Exception as e:
            return None

# Główna aplikacja
class MultilingualApp:
    """Główna klasa aplikacji"""
    
    def __init__(self):
        self.labels = Labels.get_labels()
        self.languages = Languages.get_languages()
        
        # Inicjalizacja menedżerów (bez klienta OpenAI)
        self.openai_handler = None
        self.translation_manager = None
        self.explanation_manager = None
        self.style_manager = None
        self.correction_manager = None
        self.flashcard_manager = None
        self.client = None
    
    def render_sidebar(self):
        """Renderowanie sidebar"""
        st.sidebar.title("⚙️ Ustawienia")
        
        # Wybór języka interfejsu
        lang = st.sidebar.selectbox(
            "🌐 Język interfejsu",
            ["Polski", "English", "Deutsch", "Українсьka", "Français", "Español", "العربية", "Arabski (libański dialekt)", "中文", "日本語"],
            index=0
        )
        
        # Wybór motywu
        st.sidebar.subheader("🎨 Motyw")
        bg_color = st.sidebar.radio(
            "Kolor tła",
            ["Jasny", "Ciemny"],
            index=0
        )
        
        # Informacje o aplikacji
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ O aplikacji")
        st.sidebar.markdown("""
        **Tłumacz Wielojęzyczny** to zaawansowane narzędzie do:
        - 🌍 Tłumaczenia tekstów
        - 📚 Wyjaśniania gramatyki
        - ✨ Poprawy stylistyki
        - 🔧 Korekcji błędów
        - 📖 Tworzenia fiszek
        - 🎤 Ćwiczenia wymowy
        """)
        
        # Sekcja ćwiczenia wymowy została usunięta dla kompatybilności ze Streamlit Cloud
        
        # Statystyki
        if 'request_count' not in st.session_state:
            st.session_state.request_count = 0
        
        st.sidebar.markdown(f"📊 Liczba requestów: {st.session_state.request_count}")
        
        return lang, bg_color
    
    def _extract_section(self, text: str, section_start: str) -> str:
        """Wyciąga określoną sekcję z tekstu wynikowego"""
        try:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(section_start):
                    # Wyciągnij tekst po dwukropku
                    colon_pos = line.find(':')
                    if colon_pos > 0:
                        content = line[colon_pos + 1:].strip()
                        # Jeśli następna linia nie zaczyna się od nowej sekcji, dodaj ją do treści
                        if i + 1 < len(lines) and not any(lines[i + 1].strip().startswith(x) for x in ["Wykryty język:", "Poprawiony tekst:", "Tłumaczenie na", "Transkrypcja:"]):
                            content += " " + lines[i + 1].strip()
                        return content if content else "Brak danych"
            return "Brak danych"
        except:
            return "Błąd odczytu"
    
    def apply_theme(self, bg_color: str):
        """Aplikowanie motywu"""
        if bg_color == "Ciemny":
            st.markdown("""
                <style>
                body, .stApp {
                    background-color: #0e1117 !important;
                    color: #fafafa !important;
                }
                .stTextInput > div > input,
                .stTextArea textarea,
                .stSelectbox div[role="combobox"],
                .stButton button {
                    background-color: #262730 !important;
                    color: #fafafa !important;
                    border: 1px solid #4a4a4a !important;
                }
                .stButton button:hover {
                    background-color: #4a4a4a !important;
                }
                .css-1d391kg {
                    background-color: #0e1117 !important;
                }
                </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <style>
                body, .stApp {
                    background-color: #ffffff !important;
                    color: #262730 !important;
                }
                .stTextInput > div > input,
                .stTextArea textarea,
                .stSelectbox div[role="combobox"],
                .stButton button {
                    background-color: #ffffff !important;
                    color: #262730 !important;
                    border: 1px solid #e0e0e0 !important;
                }
                .stButton button:hover {
                    background-color: #f0f0f0 !important;
                }
                .css-1d391kg {
                    background-color: #ffffff !important;
                }
                </style>
            """, unsafe_allow_html=True)
    
    def render_translation_section(self, lang: str):
        """Renderowanie sekcji tłumaczenia"""
        # Custom główny nagłówek z odpowiednim CSS
        st.markdown(f"""
        <div style="margin: 0; width: 100%; box-sizing: border-box;">
            <h1 style="margin: 0 0 30px 0; color: #1f77b4; font-size: 32px; font-weight: 700; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">{self.labels["Tłumacz wielojęzyczny"][lang]}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Inicjalizacja target_lang w session_state jeśli nie istnieje
            if "target_lang" not in st.session_state:
                st.session_state.target_lang = list(self.languages.keys())[0]
            
            target_lang = st.selectbox(
                self.labels["Wybierz język docelowy"][lang],
                list(self.languages.keys()),
                index=list(self.languages.keys()).index(st.session_state.target_lang),
                key="target_lang_selectbox"
            )
            
            # Aktualizuj session_state
            st.session_state.target_lang = target_lang
        
        with col2:
            st.markdown("")
            st.markdown("")
            if st.button(self.labels["Losowy język"][lang], key="random_lang_btn"):
                import random
                random_lang = random.choice(list(self.languages.keys()))
                st.session_state.target_lang = random_lang
                st.rerun()
        
        # Custom podnagłówek z odpowiednim CSS
        st.markdown(f"""
        <div style="margin: 0; width: 100%; box-sizing: border-box;">
            <h2 style="margin: 0 0 20px 0; color: #495057; font-size: 24px; font-weight: 600; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">{self.labels["Wpisz wiadomość do przetłumaczenia"][lang]}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Sprawdź czy jest nagrany tekst
        initial_text = ""
        if 'recorded_translation_text' in st.session_state and st.session_state.recorded_translation_text:
            initial_text = st.session_state.recorded_translation_text
        
        text = st.text_area(
            self.labels["Wprowadź tekst tutaj:"][lang],
            value=initial_text,
            height=150,
            placeholder="Wpisz tutaj tekst do przetłumaczenia...",
            key="translation_text"
        )
        
        # Sekcja rozpoznawania mowy (cloud-friendly)
        st.markdown("---")
        st.markdown(f"""
        <div style="margin: 0; width: 100%; box-sizing: border-box;">
            <h2 style="margin: 0 0 20px 0; color: #495057; font-size: 24px; font-weight: 600; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">{self.labels["Lub nagraj swoją wypowiedź"][lang]}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            mic_key = f"translation_mic_v{st.session_state.mic_widget_version}"
            mic_data = st.audio_input(self.labels["Nagraj z mikrofonu"][lang], key=mic_key)
            if mic_data is not None:
                audio_bytes = mic_data.getvalue()
                text_from_mic = self.openai_handler.transcribe_audio(audio_bytes, "mic.wav")
                if text_from_mic:
                    st.session_state.recorded_translation_text = text_from_mic
                    # Zresetuj widget przez zmianę klucza (inkrementacja wersji)
                    st.session_state.mic_widget_version += 1
                    st.success("✅ Nagrano i rozpoznano! Tekst dodano powyżej.")
                    st.rerun()
                else:
                    st.warning("⚠️ Nie udało się rozpoznać mowy.")
        with col2:
            file_key = f"translation_audio_upload_v{st.session_state.file_widget_version}"
            audio_file = st.file_uploader(
                self.labels["Wczytaj plik audio"][lang],
                type=["wav", "mp3", "m4a"],
                key=file_key
            )
            if audio_file is not None:
                uploaded_bytes = audio_file.getvalue()
                text_from_file = self.openai_handler.transcribe_audio(uploaded_bytes, audio_file.name)
                if text_from_file:
                    st.session_state.recorded_translation_text = text_from_file
                    # Zresetuj widget przez zmianę klucza (inkrementacja wersji)
                    st.session_state.file_widget_version += 1
                    st.success("✅ Wczytano i rozpoznano! Tekst dodano powyżej.")
                    st.rerun()
                else:
                    st.warning("⚠️ Nie udało się rozpoznać mowy z pliku.")
        
        st.markdown("---")
        
        # Opcje tłumaczenia
        col1, col2 = st.columns([1, 1])
        with col1:
            correct_errors = st.checkbox("🔧 Popraw błędy przed tłumaczeniem", value=False, help="Popraw błędy gramatyczne i stylistyczne w oryginalnym języku przed tłumaczeniem")
            st.session_state.correct_errors_enabled = correct_errors
        with col2:
            st.markdown("")  # Pusty element dla wyrównania
        
        # Przycisk przetłumacz w pełnej szerokości
        if st.button(
            self.labels["Przetłumacz"][lang],
            type="primary",
            use_container_width=True
        ):
            if text.strip():
                st.session_state.request_count += 1
                result = self.translation_manager.translate_text(text, target_lang, lang, correct_errors)
                
                if result:
                    st.markdown("---")
                    # Custom nagłówek wyników z odpowiednim CSS
                    st.markdown(f"""
                    <div style="margin: 0; width: 100%; box-sizing: border-box;">
                        <h3 style="margin: 0 0 20px 0; color: #1f77b4; font-size: 24px; font-weight: 700; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">✨ Wynik ({target_lang}):</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result["translation"]:
                        # Sprawdź czy to jest wynik z poprawą błędów
                        if correct_errors and ("Wykryty język:" in result["translation"] or "Poprawiony tekst:" in result["translation"]):
                            # Wyświetl w czterech kolumnach jedna pod drugą
                            
                            # Kolumna 1: Wykryty język
                            st.markdown(f"""
                            <div style="background-color: #e3f2fd; padding: 20px; border-radius: 15px; border-left: 8px solid #2196f3; margin: 0 0 20px 0; width: 100%; box-sizing: border-box; min-height: 120px;">
                                <h4 style="margin: 0 0 15px 0; color: #2196f3; font-size: 18px; font-weight: 600; text-align: left;">🔍 Wykryty język</h4>
                                <div style="font-size: 16px; line-height: 1.6; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">
                                    {self._extract_section(result["translation"], "Wykryty język:")}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Kolumna 2: Poprawiony tekst
                            st.markdown(f"""
                            <div style="background-color: #f3e5f5; padding: 20px; border-radius: 15px; border-left: 8px solid #9c27b0; margin: 0 0 20px 0; width: 100%; box-sizing: border-box; min-height: 120px;">
                                <h4 style="margin: 0 0 15px 0; color: #9c27b0; font-size: 18px; font-weight: 600; text-align: left;">✏️ Poprawiony tekst</h4>
                                <div style="font-size: 16px; line-height: 1.6; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">
                                    {self._extract_section(result["translation"], "Poprawiony tekst:")}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Kolumna 3: Tłumaczenie
                            st.markdown(f"""
                            <div style="background-color: #e8f5e8; padding: 20px; border-radius: 15px; border-left: 8px solid #4caf50; margin: 0 0 20px 0; width: 100%; box-sizing: border-box; min-height: 120px;">
                                <h4 style="margin: 0 0 15px 0; color: #4caf50; font-size: 18px; font-weight: 600; text-align: left;">🌐 Tłumaczenie</h4>
                                <div style="font-size: 16px; line-height: 1.6; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">
                                    {self._extract_section(result["translation"], "Tłumaczenie na")}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Kolumna 4: Transkrypcja
                            st.markdown(f"""
                            <div style="background-color: #fff3e0; padding: 20px; border-radius: 15px; border-left: 8px solid #ff9800; margin: 0 0 20px 0; width: 100%; box-sizing: border-box; min-height: 120px;">
                                <h4 style="margin: 0 0 15px 0; color: #ff9800; font-size: 18px; font-weight: 600; text-align: left;">🔤 Transkrypcja</h4>
                                <div style="font-size: 16px; line-height: 1.6; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">
                                    {self._extract_section(result["translation"], "Transkrypcja:")}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Standardowe wyświetlanie tłumaczenia w jednej kolumnie
                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #1f77b4; margin: 0; width: 100%; box-sizing: border-box;">
                                <h4 style="margin: 0 0 20px 0; color: #1f77b4; font-size: 20px; font-weight: 600; text-align: left;">📝 Tłumaczenie:</h4>
                                <div style="font-size: 18px; line-height: 1.8; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">{result['translation']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Nie znaleziono tłumaczenia w odpowiedzi.")
                    
                    # Transkrypcja jest już wbudowana w tłumaczenie, więc nie wyświetlamy jej osobno
                    
                    # Generowanie audio
                    if result["translation"]:
                        st.markdown("---")
                        # Audio w pełnej szerokości
                        audio_content = self.translation_manager.generate_audio(result["translation"])
                        if audio_content:
                            # Wyświetl audio w lepszym formacie
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 2px solid #e9ecef; margin: 10px 0; width: 100%; box-sizing: border-box;">
                                <h4 style="margin: 0 0 15px 0; color: #495057; font-size: 18px; font-weight: 600; text-align: left;">🔊 Odsłuchaj tłumaczenie</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            st.audio(audio_content, format="audio/mp3")
                            

            else:
                st.warning("⚠️ Wpisz tekst do przetłumaczenia.")
    
    def render_explanation_section(self, lang: str):
        """Renderowanie sekcji wyjaśnień"""
        st.header(self.labels["Wyjaśnienia słów i gramatyki"][lang])
        
        explain_text = st.text_area(
            self.labels["Wpisz zdanie lub tekst do wyjaśnienia:"][lang],
            height=120,
            placeholder="Wpisz tutaj tekst do wyjaśnienia...",
            key="explanation_text"
        )
        
        if st.button(
            self.labels["Wyjaśnij słowa i gramatykę"][lang],
            type="secondary",
            use_container_width=True
        ):
            if explain_text:
                st.session_state.request_count += 1
                explanation = self.explanation_manager.explain_text(explain_text)
                
                if explanation:
                    st.markdown("---")
                    # Wyświetl wyjaśnienia w lepszym formacie
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #28a745; margin: 0; width: 100%; box-sizing: border-box;">
                        <h4 style="margin: 0 0 20px 0; color: #28a745; font-size: 20px; font-weight: 600; text-align: left;">📚 Wyjaśnienia:</h4>
                        <div style="font-size: 18px; line-height: 1.8; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">{explanation}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Wpisz tekst do wyjaśnienia.")
    
    def render_style_section(self, lang: str):
        # Jeśli włączona jest opcja poprawy błędów przed tłumaczeniem, pokazujemy sekcję stylistyki tylko gdy użytkownik faktycznie jej potrzebuje
        # (nie ukrywamy twardo, ale zostawiamy jasny podtytuł)
        """Renderowanie sekcji stylistyki"""
        st.header(self.labels["Ładna wersja wypowiedzi – poprawa stylistyki"][lang])
        st.caption("Nie tłumaczy — tylko poprawa stylu i gramatyki w tym samym języku.")
        
        style_text = st.text_area(
            self.labels["Wpisz tekst do poprawy stylistycznej:"][lang],
            height=120,
            placeholder="Wpisz tutaj tekst do poprawy...",
            key="style_text"
        )
        
        if st.button(
            self.labels["Popraw stylistykę i wygeneruj ładną wersję"][lang],
            type="secondary",
            use_container_width=True
        ):
            if style_text:
                st.session_state.request_count += 1
                nice_version = self.style_manager.improve_style(style_text)
                
                if nice_version:
                    st.markdown("---")
                    # Wyświetl ładną wersję w lepszym formacie
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #ffc107; margin: 0; width: 100%; box-sizing: border-box;">
                        <h4 style="margin: 0 0 20px 0; color: #ffc107; font-size: 20px; font-weight: 600; text-align: left;">✨ Ładna wersja wypowiedzi:</h4>
                        <div style="font-size: 18px; line-height: 1.8; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">{nice_version}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Wpisz tekst do poprawy stylistycznej.")
    

    
    def render_flashcard_section(self, lang: str):
        """Renderowanie sekcji fiszek"""
        st.header(self.labels["Fiszki ze słówek do nauki"][lang])
        
        flashcard_text = st.text_area(
            self.labels["Wpisz tekst, z którego chcesz wygenerować fiszki:"][lang],
            height=120,
            placeholder="Wpisz tutaj tekst do wygenerowania fiszek...",
            key="flashcard_text"
        )
        
        if st.button(
            self.labels["Wygeneruj fiszki"][lang],
            type="secondary",
            use_container_width=True
        ):
            if flashcard_text:
                st.session_state.request_count += 1
                flashcards_data = self.flashcard_manager.generate_flashcards(flashcard_text)
                
                if flashcards_data and "flashcards" in flashcards_data:
                    st.markdown("---")
                    # Wyświetl fiszki w lepszym formacie
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #6f42c1; margin: 0; width: 100%; box-sizing: border-box;">
                        <h4 style="margin: 0 0 20px 0; color: #6f42c1; font-size: 20px; font-weight: 600; text-align: left;">📖 Wygenerowane fiszki:</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Sprawdź czy to nie są fiszki z błędami
                    if len(flashcards_data["flashcards"]) == 1 and flashcards_data["flashcards"][0].get("word", "").startswith("Błąd"):
                        st.error("❌ Wystąpił błąd podczas generowania fiszek. Spróbuj ponownie.")
                        st.info("💡 **Wskazówka:** Upewnij się, że tekst jest w języku, który chcesz przetłumaczyć.")
                        return
                    
                    # Wyświetl fiszki w ładnym formacie
                    for i, card in enumerate(flashcards_data["flashcards"], 1):
                        with st.expander(f"🃏 Fiszka {i}: {card.get('word', 'Brak słówka')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**🔤 Słówko:** {card.get('word', 'Brak')}")
                                st.markdown(f"**📝 Definicja:** {card.get('definition', 'Brak')}")
                            with col2:
                                st.markdown(f"**💡 Przykład:** {card.get('example', 'Brak')}")
                    
                    # Generuj obrazy fiszek
                    st.markdown("---")
                    # Wyświetl nagłówek w lepszym formacie
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #6f42c1; margin: 0; width: 100%; box-sizing: border-box;">
                        <h4 style="margin: 0 0 20px 0; color: #6f42c1; font-size: 20px; font-weight: 600; text-align: left;">🖼️ Pobierz fiszki do wydruku</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Wybór formatu
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        format_choice = st.selectbox(
                            "📁 Wybierz format:",
                            ["PNG (najlepsza jakość)", "JPG (mniejszy rozmiar)", "PDF (do drukowania)"],
                            index=0
                        )
                    
                    with col2:
                        quality_choice = st.selectbox(
                            "⭐ Jakość:",
                            ["Wysoka", "Średnia", "Niska"],
                            index=0
                        )
                    
                    with col3:
                        size_choice = st.selectbox(
                            "📏 Rozmiar fiszek:",
                            ["Duże (800×600)", "Średnie (600×450)", "Małe (400×300)"],
                            index=0
                        )
                    
                    # Generowanie obrazu
                    image_data = self.flashcard_manager.generate_images(flashcards_data, size_choice, format_choice, quality_choice)
                    
                    if image_data:
                        st.success("✅ Obraz został wygenerowany pomyślnie!")
                        
                        # Podgląd obrazu
                        st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #6f42c1; margin: 0; width: 100%; box-sizing: border-box;">
                            <h4 style="margin: 0 0 20px 0; color: #6f42c1; font-size: 20px; font-weight: 600; text-align: left;">👀 Podgląd fiszek:</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(image_data, caption="Podgląd wygenerowanych fiszek", use_container_width=True)
                        
                        # Przyciski pobierania
                        col1, col2 = st.columns(2)
                        with col1:
                            # Określenie formatu i rozszerzenia pliku
                            if "JPG" in format_choice:
                                file_extension = "jpg"
                                mime_type = "image/jpeg"
                            else:
                                file_extension = "png"
                                mime_type = "image/png"
                            
                            st.download_button(
                                label="📥 Pobierz fiszki",
                                data=image_data,
                                file_name=f"fiszki_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                                mime=mime_type,
                                use_container_width=True,
                                type="primary"
                            )
                        

                        
                        # Szczegółowe instrukcje
                        with st.expander("📋 📏 Szczegółowe instrukcje wycinania"):
                            st.markdown("""
                            ### ✂️ **Jak wyciąć i przygotować fiszki:**
                            
                            **📏 Wymiary fiszek:** 
                            - **Duże:** 800×600 pikseli (≈ 21×16 cm)
                            - **Średnie:** 600×450 pikseli (≈ 16×12 cm)  
                            - **Małe:** 400×300 pikseli (≈ 10×8 cm)
                            
                            **🖨️ Drukowanie:**
                            1. Użyj papieru A4 (210×297 mm)
                            2. Ustaw jakość drukowania na "Wysoką"
                            3. Wyłącz skalowanie - drukuj w 100%
                            
                            **✂️ Wycinanie:**
                            1. Wytnij każdą fiszkę wzdłuż niebieskiej ramki
                            2. Złóż na pół wzdłuż pomarańczowej linii
                            3. Słówko będzie na przodzie, definicja na tyle
                            
                            **💎 Laminowanie (opcjonalne):**
                            - Użyj folii laminującej 125 mikronów
                            - Temperatura: 130-140°C
                            - Czas: 30-60 sekund
                            
                            **🎯 Wskazówki:**
                            - Użyj ostrych nożyczek lub noża introligatorskiego
                            - Możesz użyć perforatora do łatwiejszego składania
                            - Przechowuj w pudełku lub teczce
                            """)
                        
                        st.info("💡 **Szybkie instrukcje:** Wydrukuj obraz, wytnij fiszki wzdłuż linii i złóż na pół. Możesz zalaminować dla trwałości!")
                    else:
                        st.error("❌ Błąd generowania obrazu")
                else:
                    st.warning("Nie udało się wygenerować fiszek.")
            else:
                st.warning("Wpisz tekst do wygenerowania fiszek.")
    
    def render_footer(self):
        """Renderowanie stopki"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>🌍 <strong>Tłumacz Wielojęzyczny</strong> - Twoje narzędzie do nauki języków</p>
            <p>Made with ❤️ using Streamlit & OpenAI</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metoda render_pronunciation_practice_section została usunięta dla kompatybilności ze Streamlit Cloud
    
    # Metody związane z ćwiczeniem wymowy zostały usunięte dla kompatybilności ze Streamlit Cloud
    
    def run(self):
        """Uruchomienie aplikacji"""
        # Inicjalizacja klienta OpenAI
        api_key = api_key_input()
        if not api_key:
            st.error("❌ Nie można zainicjalizować klienta OpenAI. Sprawdź klucz API.")
            st.stop()
        
        self.client = get_openai_client(api_key)
        if not self.client:
            st.error("❌ Nie można zainicjalizować klienta OpenAI. Sprawdź klucz API.")
            st.stop()
        
        # Inicjalizacja menedżerów
        self.openai_handler = OpenAIHandler(self.client)
        self.translation_manager = TranslationManager(self.openai_handler)
        self.explanation_manager = ExplanationManager(self.openai_handler)
        self.style_manager = StyleManager(self.openai_handler)
        self.correction_manager = CorrectionManager(self.openai_handler)
        self.flashcard_manager = FlashcardManager(self.openai_handler)
        
        # Renderuj sidebar
        lang, bg_color = self.render_sidebar()
        
        # Wyświetl statystyki użycia API
        display_usage_stats()
        
        # Aplikuj motyw
        self.apply_theme(bg_color)
        
        # Sekcje aplikacji
        self.render_translation_section(lang)
        st.markdown("---")
        
        self.render_explanation_section(lang)
        st.markdown("---")
        
        self.render_style_section(lang)
        st.markdown("---")
        
        self.render_flashcard_section(lang)
        
        # Stopka
        self.render_footer()

# Uruchomienie aplikacji
if __name__ == "__main__":
    # Inicjalizacja stanu sesji przed uruchomieniem aplikacji
    init_session_state()
    app = MultilingualApp()
    app.run()