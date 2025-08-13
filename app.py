import streamlit as st
import streamlit.components.v1 as components
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
try:
    import speech_recognition as sr  # type: ignore
except Exception:
    sr = None  # type: ignore

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
    # Wersjonowanie pola tekstowego, aby umożliwić czyszczenie bez modyfikacji st.session_state po renderze
    if 'translation_text_version' not in st.session_state:
        st.session_state.translation_text_version = 0
    # Pronunciation practice state
    if 'practice_text' not in st.session_state:
        st.session_state.practice_text = ""
    if 'practice_mic_version' not in st.session_state:
        st.session_state.practice_mic_version = 0
    # Wersjonowanie kluczy widgetów audio, aby uniknąć ponownego przetwarzania po rerun
    if 'mic_widget_version' not in st.session_state:
        st.session_state.mic_widget_version = 0
    if 'file_widget_version' not in st.session_state:
        st.session_state.file_widget_version = 0
    # Zmienne związane z ćwiczeniem wymowy zostały usunięte dla kompatybilności ze Streamlit Cloud
    # Ustawienia startowe (setup gate)
    if 'setup_done' not in st.session_state:
        st.session_state.setup_done = False
    if 'interface_lang' not in st.session_state:
        st.session_state.interface_lang = "Polski"




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
def display_usage_stats(lang: str, labels: Dict[str, Dict[str, str]]):
    """Wyświetl statystyki użycia API (i18n)"""
    with st.sidebar:
        st.markdown(f"### {labels['API stats'][lang]}")
        
        # Aktualne statystyki (ułożone pionowo, by nie poszerzać sidebara)
        st.metric(labels["Total tokens"][lang], f"{st.session_state.total_tokens:,}")
        st.metric(labels["Total cost"][lang], f"${st.session_state.total_cost:.4f}")
        
        # Szczegółowe statystyki
        if st.session_state.token_history:
            st.markdown(f"#### {labels['Last usage'][lang]}")
            latest = st.session_state.token_history[-1]
            # Oblicz koszt dla ostatniego użycia
            latest_cost = calculate_cost(latest['model'], latest['input_tokens'], latest['output_tokens'])
            st.info(f"""
            **{labels['Model label'][lang]}** {latest['model']}  
            **{labels['Input tokens'][lang]}** {latest['input_tokens']:,}  
            **{labels['Output tokens'][lang]}** {latest['output_tokens']:,}  
            **{labels['Cost label'][lang]}** ${latest_cost:.4f}
            """)
        
        # Historia kosztów
        if st.session_state.cost_history:
            with st.expander(labels["Cost history"][lang]):
                for entry in reversed(st.session_state.cost_history[-10:]):  # Ostatnie 10
                    st.text(f"{entry['timestamp']}: ${entry['cost']:.4f} ({entry['model']})")
        
        # Reset statystyk
        if st.button(labels["Reset stats"][lang], use_container_width=True):
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

# Klasy pomocnicze do i18n z automatycznym fallbackiem tłumaczeń
class LabelsEntry:
    """Pojedyncza etykieta wielojęzyczna z automatycznym fallbackiem tłumaczenia."""

    def __init__(self, key: str, language_to_text: Dict[str, str]):
        self.key = key
        self.language_to_text = language_to_text

    def _get_base_text(self) -> str:
        # Preferuj English, następnie Polski, potem dowolną istniejącą wartość
        if "English" in self.language_to_text and self.language_to_text["English"]:
            return self.language_to_text["English"]
        if "Polski" in self.language_to_text and self.language_to_text["Polski"]:
            return self.language_to_text["Polski"]
        for _lang, _text in self.language_to_text.items():
            if _text:
                return _text
        return self.key

    def _translate_on_the_fly(self, target_lang: str) -> str:
        import streamlit as st  # lokalny import, aby uniknąć cykli
        cache = st.session_state.setdefault("i18n_cache", {})
        lang_cache = cache.setdefault(target_lang, {})
        if self.key in lang_cache and lang_cache[self.key]:
            translated = lang_cache[self.key]
            self.language_to_text[target_lang] = translated
            return translated

        base_text = self._get_base_text()

        translator = Labels._translator
        if translator is None:
            return base_text

        prompt = (
            f"Translate the following UI label to {target_lang}. Keep any emoji and punctuation EXACTLY as in the source. "
            f"Return ONLY the translated text, no quotes, no extra words.\nLabel: {base_text}"
        )
        messages = [
            {"role": "system", "content": "You are a professional UI localizer. Keep emoji and casing exactly."},
            {"role": "user", "content": prompt},
        ]
        try:
            translated = translator(messages) or base_text
            translated = translated.strip()
        except Exception:
            translated = base_text

        lang_cache[self.key] = translated
        self.language_to_text[target_lang] = translated
        return translated

    def __getitem__(self, target_lang: str) -> str:
        if target_lang in self.language_to_text and self.language_to_text[target_lang]:
            return self.language_to_text[target_lang]
        return self._translate_on_the_fly(target_lang)

    def get(self, target_lang: str, default: str = "") -> str:
        try:
            return self.__getitem__(target_lang)
        except Exception:
            return default


class LabelsStore:
    """Kontener na wszystkie etykiety z API podobnym do dict, z auto-fallbackiem."""

    def __init__(self, data: Dict[str, Dict[str, str]]):
        self._data = data
        self._entries: Dict[str, LabelsEntry] = {}

    def __getitem__(self, key: str) -> LabelsEntry:
        if key not in self._entries:
            self._entries[key] = LabelsEntry(key, self._data.get(key, {}))
        return self._entries[key]

    def get(self, key: str, default: Optional[Dict[str, str]] = None) -> LabelsEntry:
        if key not in self._data:
            self._data[key] = {}
        return self.__getitem__(key)


# Słownik etykiet z lepszą organizacją
class Labels:
    """Zarządzanie etykietami w różnych językach"""

    _translator = None  # ustawiane w runtime

    @staticmethod
    def set_translator(openai_handler) -> None:
        if openai_handler is None:
            Labels._translator = None
            return

        def _translate_messages(messages):
            return openai_handler.make_request(messages)

        Labels._translator = _translate_messages

    @staticmethod
    @st.cache_data(ttl=7200)  # Cache na 2 godziny
    def wrap_labels(data: Dict[str, Dict[str, str]]) -> LabelsStore:
        return LabelsStore(data)
    
    @staticmethod
    def get_labels() -> LabelsStore:
        return Labels.wrap_labels({
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
            "Detected language": {
                "Polski": "🔍 Wykryty język",
                "English": "🔍 Detected language",
                "Deutsch": "🔍 Erkannte Sprache",
                "Українська": "🔍 Визначена мова",
                "Français": "🔍 Langue détectée",
                "Español": "🔍 Idioma detectado",
                "العربية": "🔍 اللغة المكتشفة",
                "Arabski (libański dialekt)": "🔍 اللغة المكتشفة",
                "中文": "🔍 检测到的语言",
                "日本語": "🔍 検出された言語"
            },
            "Corrected text": {
                "Polski": "✏️ Poprawiony tekst",
                "English": "✏️ Corrected text",
                "Deutsch": "✏️ Korrigierter Text",
                "Українська": "✏️ Виправлений текст",
                "Français": "✏️ Texte corrigé",
                "Español": "✏️ Texto corregido",
                "العربية": "✏️ النص المصحح",
                "Arabski (libański dialekt)": "✏️ النص المصحح",
                "中文": "✏️ 修正后的文本",
                "日本語": "✏️ 修正されたテキスト"
            },
            "Transcription": {
                "Polski": "🔤 Transkrypcja",
                "English": "🔤 Transcription",
                "Deutsch": "🔤 Transkription",
                "Українська": "🔤 Транскрипція",
                "Français": "🔤 Transcription",
                "Español": "🔤 Transcripción",
                "العربية": "🔤 النسخ الصوتي",
                "Arabski (libański dialekt)": "🔤 التفريغ الصوتي",
                "中文": "🔤 转写",
                "日本語": "🔤 転写"
            },
            "Generate image": {
                "Polski": "🖼️ Wygeneruj obraz",
                "English": "🖼️ Generate image",
                "Deutsch": "🖼️ Bild generieren",
                "Українська": "🖼️ Згенерувати зображення",
                "Français": "🖼️ Générer l'image",
                "Español": "🖼️ Generar imagen",
                "العربية": "🖼️ ولّد الصورة",
                "Arabski (libański dialekt)": "🖼️ ولّد الصورة",
                "中文": "🖼️ 生成图像",
                "日本語": "🖼️ 画像を生成"
            },
            "Quick tips": {
                "Polski": "💡 **Szybkie instrukcje:** Wydrukuj obraz, wytnij fiszki wzdłuż linii i złóż na pół. Możesz zalaminować dla trwałości!",
                "English": "💡 **Quick tips:** Print the image, cut along the lines, and fold in half. Laminating increases durability!",
                "Deutsch": "💡 **Schnelle Tipps:** Bild drucken, entlang der Linien schneiden und in der Mitte falten. Laminieren erhöht die Haltbarkeit!",
                "Українська": "💡 **Швидкі поради:** Надрукуйте зображення, виріжте по лініях і складіть навпіл. Ламінування підвищує міцність!",
                "Français": "💡 **Conseils rapides :** Imprimez l'image, découpez le long des lignes et pliez en deux. Le plastifiage augmente la durabilité !",
                "Español": "💡 **Consejos rápidos:** Imprime la imagen, corta por las líneas y dóblala por la mitad. ¡Laminar aumenta la durabilidad!",
                "العربية": "💡 **نصائح سريعة:** اطبع الصورة، قصّ على طول الخطوط واطوِ إلى النصف. التغليف يزيد المتانة!",
                "Arabski (libański dialekt)": "💡 **نصايح سريعة:** اطبع الصورة، قصّ عالخطوط واطوّيها بالنص. التغليف بيزيد المتانة!",
                "中文": "💡 **快速提示：** 打印图片，沿线裁切并对折。覆膜可提高耐用性！",
                "日本語": "💡 **クイックヒント：** 画像を印刷し、線に沿って切って二つ折りにします。ラミネートすると耐久性が上がります！"
            },
            "Image generated ok": {
                "Polski": "✅ Obraz został wygenerowany pomyślnie!",
                "English": "✅ Image generated successfully!",
                "Deutsch": "✅ Bild erfolgreich generiert!",
                "Українська": "✅ Зображення успішно згенеровано!",
                "Français": "✅ Image générée avec succès !",
                "Español": "✅ ¡Imagen generada con éxito!",
                "العربية": "✅ تم إنشاء الصورة بنجاح!",
                "Arabski (libański dialekt)": "✅ الصورة تولّدت بنجاح!",
                "中文": "✅ 图像生成成功！",
                "日本語": "✅ 画像が正常に生成されました！"
            },
            "Flashcards preview": {
                "Polski": "👀 Podgląd fiszek:",
                "English": "👀 Flashcards preview:",
                "Deutsch": "👀 Vorschau der Karteikarten:",
                "Українська": "👀 Попередній перегляд карток:",
                "Français": "👀 Aperçu des fiches :",
                "Español": "👀 Vista previa de las tarjetas:",
                "العربية": "👀 معاينة البطاقات:",
                "Arabski (libański dialekt)": "👀 معاينة البطاقات:",
                "中文": "👀 卡片预览：",
                "日本語": "👀 フラッシュカードのプレビュー："
            },
            "Flashcard expander title": {
                "Polski": "Fiszka",
                "English": "Flashcard",
                "Deutsch": "Karte",
                "Українська": "Картка",
                "Français": "Fiche",
                "Español": "Tarjeta",
                "العربية": "بطاقة",
                "Arabski (libański dialekt)": "بطاقة",
                "中文": "卡片",
                "日本語": "カード"
            },
            "Missing - word": {
                "Polski": "Brak",
                "English": "N/A",
                "Deutsch": "k.A.",
                "Українська": "Н/Д",
                "Français": "N/A",
                "Español": "N/D",
                "العربية": "غير متوفر",
                "Arabski (libański dialekt)": "مش متوفر",
                "中文": "无",
                "日本語": "なし"
            },
            "Cutting instructions - expander": {
                "Polski": "📋 📏 Szczegółowe instrukcje wycinania",
                "English": "📋 📏 Detailed cutting instructions",
                "Deutsch": "📋 📏 Detaillierte Ausschneideanleitung",
                "Українська": "📋 📏 Детальні інструкції з вирізання",
                "Français": "📋 📏 Instructions détaillées de découpe",
                "Español": "📋 📏 Instrucciones detalladas de corte",
                "العربية": "📋 📏 تعليمات مفصلة للقص",
                "Arabski (libański dialekt)": "📋 📏 تعليمات قص مفصلة",
                "中文": "📋 📏 详细裁剪说明",
                "日本語": "📋 📏 詳細な切り抜き手順"
            },
            "Cutting instructions - content": {
                "Polski": """
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
                            """,
                "English": """
                            ### ✂️ **How to cut and prepare flashcards:**
                            
                            **📏 Flashcard sizes:** 
                            - **Large:** 800×600 px (≈ 21×16 cm)
                            - **Medium:** 600×450 px (≈ 16×12 cm)  
                            - **Small:** 400×300 px (≈ 10×8 cm)
                            
                            **🖨️ Printing:**
                            1. Use A4 paper (210×297 mm)
                            2. Set print quality to "High"
                            3. Disable scaling – print at 100%
                            
                            **✂️ Cutting:**
                            1. Cut each card along the blue border
                            2. Fold in half along the orange line
                            3. Word on the front, definition on the back
                            
                            **💎 Laminating (optional):**
                            - Use 125-micron laminating pouches
                            - Temperature: 130–140°C
                            - Time: 30–60 seconds
                            
                            **🎯 Tips:**
                            - Use sharp scissors or a craft knife
                            - You can score the fold line for easier folding
                            - Store in a box or folder
                            """,
                "Deutsch": """
                            ### ✂️ **Karteikarten ausschneiden und vorbereiten:**
                            
                            **📏 Kartengrößen:** 
                            - **Groß:** 800×600 px (≈ 21×16 cm)
                            - **Mittel:** 600×450 px (≈ 16×12 cm)  
                            - **Klein:** 400×300 px (≈ 10×8 cm)
                            
                            **🖨️ Drucken:**
                            1. A4-Papier verwenden (210×297 mm)
                            2. Druckqualität auf „Hoch" stellen
                            3. Skalierung deaktivieren – in 100% drucken
                            
                            **✂️ Schneiden:**
                            1. Jede Karte entlang des blauen Randes ausschneiden
                            2. Entlang der orangefarbenen Linie in der Mitte falten
                            3. Wort vorne, Definition hinten
                            
                            **💎 Laminieren (optional):**
                            - Laminierfolien 125 µm verwenden
                            - Temperatur: 130–140°C
                            - Zeit: 30–60 Sekunden
                            
                            **🎯 Tipps:**
                            - Scharfe Schere oder Bastelmesser verwenden
                            - Falzlinie rillen, um das Falten zu erleichtern
                            - In einer Box oder Mappe aufbewahren
                            """,
                "Español": """
                            ### ✂️ **Cómo recortar y preparar las tarjetas:**
                            
                            **📏 Tamaños de tarjeta:** 
                            - **Grandes:** 800×600 px (≈ 21×16 cm)
                            - **Medianas:** 600×450 px (≈ 16×12 cm)  
                            - **Pequeñas:** 400×300 px (≈ 10×8 cm)
                            
                            **🖨️ Impresión:**
                            1. Usa papel A4 (210×297 mm)
                            2. Configura la calidad de impresión en "Alta"
                            3. Desactiva el escalado – imprime al 100%
                            
                            **✂️ Corte:**
                            1. Recorta cada tarjeta a lo largo del borde azul
                            2. Dóblala por la línea naranja
                            3. Palabra al frente, definición detrás
                            
                            **💎 Plastificado (opcional):**
                            - Utiliza fundas de 125 micras
                            - Temperatura: 130–140°C
                            - Tiempo: 30–60 segundos
                            
                            **🎯 Consejos:**
                            - Usa tijeras afiladas o cúter
                            - Marca la línea de pliegue para doblar más fácil
                            - Guarda en una caja o carpeta
                            """,
                "Français": """
                            ### ✂️ **Comment découper et préparer les fiches :**
                            
                            **📏 Tailles des fiches :** 
                            - **Grandes :** 800×600 px (≈ 21×16 cm)
                            - **Moyennes :** 600×450 px (≈ 16×12 cm)  
                            - **Petites :** 400×300 px (≈ 10×8 cm)
                            
                            **🖨️ Impression :**
                            1. Utilisez du papier A4 (210×297 mm)
                            2. Réglez la qualité d'impression sur « Élevée »
                            3. Désactivez l'échelle – imprimez à 100 %
                            
                            **✂️ Découpe :**
                            1. Découpez chaque fiche le long du bord bleu
                            2. Pliez en deux le long de la ligne orange
                            3. Mot au recto, définition au verso
                            
                            **💎 Plastification (optionnel) :**
                            - Utilisez des pochettes 125 microns
                            - Température : 130–140°C
                            - Temps : 30–60 secondes
                            
                            **🎯 Conseils :**
                            - Utilisez des ciseaux bien affûtés ou un cutter
                            - Marquez le pli pour faciliter le pliage
                            - Rangez dans une boîte ou un classeur
                            """,
                "Українська": """
                            ### ✂️ **Як вирізати та підготувати картки:**
                            
                            **📏 Розміри карток:** 
                            - **Великі:** 800×600 px (≈ 21×16 см)
                            - **Середні:** 600×450 px (≈ 16×12 см)  
                            - **Малі:** 400×300 px (≈ 10×8 см)
                            
                            **🖨️ Друк:**
                            1. Використовуйте папір A4 (210×297 мм)
                            2. Установіть якість друку «Висока»
                            3. Вимкніть масштабування – друкуйте у 100%
                            
                            **✂️ Вирізання:**
                            1. Виріжте кожну картку по синій рамці
                            2. Складіть навпіл по помаранчевій лінії
                            3. Слово спереду, визначення ззаду
                            
                            **💎 Ламінування (за бажанням):**
                            - Використовуйте плівку 125 мікрон
                            - Температура: 130–140°C
                            - Час: 30–60 секунд
                            
                            **🎯 Поради:**
                            - Використовуйте гострі ножиці або канцелярський ніж
                            - Намічайте лінію згину для зручності
                            - Зберігайте у коробці або теці
                            """,
                "العربية": """
                            ### ✂️ **كيفية قصّ وتحضير البطاقات:**
                            
                            **📏 أحجام البطاقات:** 
                            - **كبيرة:** ‎800×600‎ بكسل (≈ ‎21×16‎ سم)
                            - **متوسطة:** ‎600×450‎ بكسل (≈ ‎16×12‎ سم)  
                            - **صغيرة:** ‎400×300‎ بكسل (≈ ‎10×8‎ سم)
                            
                            **🖨️ الطباعة:**
                            1. استخدم ورق A4 ‏(210×297 مم)
                            2. اضبط جودة الطباعة على «عالية»
                            3. عطّل التحجيم – اطبع بنسبة ‎100%‎
                            
                            **✂️ القصّ:**
                            1. اقصص كل بطاقة على طول الإطار الأزرق
                            2. اطوِ البطاقة على طول الخط البرتقالي
                            3. الكلمة في الأمام، والتعريف في الخلف
                            
                            **💎 التغطيس (اختياري):**
                            - استخدم أظرف تغليف 125 ميكرون
                            - الحرارة: ‎140–130°م
                            - الزمن: ‎60–30 ثانية
                            
                            **🎯 نصائح:**
                            - استخدم مقصًا حادًا أو سكينًا حرفيًا
                            - يمكن وضع خط تكسيري لتسهيل الطي
                            - خزّنها في صندوق أو ملف
                            """,
                "Arabski (libański dialekt)": """
                            ### ✂️ **كيف تقصّ وتجهّز البطاقات:**
                            
                            **📏 أحجام البطاقات:** 
                            - **كبيرة:** ‎800×600‎ بكسل
                            - **متوسطة:** ‎600×450‎ بكسل  
                            - **صغيرة:** ‎400×300‎ بكسل
                            
                            **🖨️ الطباعة:**
                            1. ورق A4
                            2. الجودة «عالية»
                            3. اطبع ‎100%‎ بدون تكبير
                            
                            **✂️ القصّ:**
                            1. قصّ على الخط الأزرق
                            2. اطوي على الخط البرتقالي
                            3. الكلمة قدّام والتعريف ورا
                            
                            **💎 تلبيس (اختياري):**
                            - أكياس 125 ميكرون
                            - حرارة 130–140°
                            - وقت 30–60 ثانية
                            
                            **🎯 نصايح:**
                            - مقصّ حاد أو كتر
                            - اعمل خط طي لتسهيل الطوي
                            - خزّنها بعلبة أو ملف
                            """,
                "中文": """
                            ### ✂️ **如何裁切并准备学习卡片：**
                            
                            **📏 卡片尺寸：** 
                            - **大：** 800×600 像素（≈ 21×16 厘米）
                            - **中：** 600×450 像素（≈ 16×12 厘米）  
                            - **小：** 400×300 像素（≈ 10×8 厘米）
                            
                            **🖨️ 打印：**
                            1. 使用 A4 纸（210×297 毫米）
                            2. 打印质量设为"高"
                            3. 关闭缩放 — 按 100% 比例打印
                            
                            **✂️ 裁切：**
                            1. 沿蓝色边框裁切每张卡片
                            2. 沿橙色线对折
                            3. 正面为单词，背面为释义
                            
                            **💎 覆膜（可选）：**
                            - 使用 125 微米覆膜
                            - 温度：130–140°C
                            - 时间：30–60 秒
                            
                            **🎯 小贴士：**
                            - 使用锋利的剪刀或美工刀
                            - 可先压折线以便折叠
                            - 存放在盒子或文件夹中
                            """,
                "日本語": """
                            ### ✂️ **フラッシュカードの切り出しと準備方法：**
                            
                            **📏 カードサイズ：** 
                            - **大：** 800×600 px（約 21×16 cm）
                            - **中：** 600×450 px（約 16×12 cm）  
                            - **小：** 400×300 px（約 10×8 cm）
                            
                            **🖨️ 印刷：**
                            1. A4 用紙（210×297 mm）を使用
                            2. 印刷品質を「高」に設定
                            3. 拡大縮小を無効にし、100% で印刷
                            
                            **✂️ カット：**
                            1. 青い枠に沿って各カードを切り取る
                            2. オレンジの線に沿って二つ折り
                            3. 表に単語、裏に定義
                            
                            **💎 ラミネート（任意）：**
                            - 125 ミクロンのラミネートフィルム
                            - 温度：130–140°C
                            - 時間：30–60 秒
                            
                            **🎯 コツ：**
                            - 切れ味の良いハサミやカッターを使用
                            - 折りやすいように折り目をスジ入れ
                            - 箱やフォルダーに保管
                            """
            },
            "Select format": {
                "Polski": "📁 Wybierz format:",
                "English": "📁 Select format:",
                "Deutsch": "📁 Format auswählen:",
                "Українська": "📁 Виберіть формат:",
                "Français": "📁 Sélectionner le format :",
                "Español": "📁 Seleccionar formato:",
                "العربية": "📁 اختر التنسيق:",
                "Arabski (libański dialekt)": "📁 اختر الفورمات:",
                "中文": "📁 选择格式：",
                "日本語": "📁 形式を選択："
            },
            "Format - PNG best": {
                "Polski": "PNG (najlepsza jakość)",
                "English": "PNG (best quality)",
                "Deutsch": "PNG (beste Qualität)",
                "Українська": "PNG (найкраща якість)",
                "Français": "PNG (meilleure qualité)",
                "Español": "PNG (mejor calidad)",
                "العربية": "PNG (أفضل جودة)",
                "Arabski (libański dialekt)": "PNG (أفضل جودة)",
                "中文": "PNG（最佳质量）",
                "日本語": "PNG（最高品質）"
            },
            "Format - JPG smaller": {
                "Polski": "JPG (mniejszy rozmiar)",
                "English": "JPG (smaller size)",
                "Deutsch": "JPG (kleinere Größe)",
                "Українська": "JPG (менший розмір)",
                "Français": "JPG (taille plus petite)",
                "Español": "JPG (tamaño más pequeño)",
                "العربية": "JPG (حجم أصغر)",
                "Arabski (libański dialekt)": "JPG (حجم أصغر)",
                "中文": "JPG（更小体积）",
                "日本語": "JPG（小さいサイズ）"
            },
            "Format - PDF print": {
                "Polski": "PDF (do drukowania)",
                "English": "PDF (for printing)",
                "Deutsch": "PDF (zum Drucken)",
                "Українська": "PDF (для друку)",
                "Français": "PDF (pour impression)",
                "Español": "PDF (para imprimir)",
                "العربية": "PDF (للطباعة)",
                "Arabski (libański dialekt)": "PDF (للطباعة)",
                "中文": "PDF（打印用）",
                "日本語": "PDF（印刷用）"
            },
            "Quality": {
                "Polski": "⭐ Jakość:",
                "English": "⭐ Quality:",
                "Deutsch": "⭐ Qualität:",
                "Українська": "⭐ Якість:",
                "Français": "⭐ Qualité :",
                "Español": "⭐ Calidad:",
                "العربية": "⭐ الجودة:",
                "Arabski (libański dialekt)": "⭐ الجودة:",
                "中文": "⭐ 质量：",
                "日本語": "⭐ 品質："
            },
            "Quality - High": {
                "Polski": "Wysoka",
                "English": "High",
                "Deutsch": "Hoch",
                "Українська": "Висока",
                "Français": "Élevée",
                "Español": "Alta",
                "العربية": "عالية",
                "Arabski (libański dialekt)": "عالية",
                "中文": "高",
                "日本語": "高"
            },
            "Quality - Medium": {
                "Polski": "Średnia",
                "English": "Medium",
                "Deutsch": "Mittel",
                "Українська": "Середня",
                "Français": "Moyenne",
                "Español": "Media",
                "العربية": "متوسطة",
                "Arabski (libański dialekt)": "متوسطة",
                "中文": "中",
                "日本語": "中"
            },
            "Quality - Low": {
                "Polski": "Niska",
                "English": "Low",
                "Deutsch": "Niedrig",
                "Українська": "Низька",
                "Français": "Faible",
                "Español": "Baja",
                "العربية": "منخفضة",
                "Arabski (libański dialekt)": "منخفضة",
                "中文": "低",
                "日本語": "低"
            },
            "Flashcard size": {
                "Polski": "📏 Rozmiar fiszek:",
                "English": "📏 Flashcard size:",
                "Deutsch": "📏 Kartengröße:",
                "Українська": "📏 Розмір карток:",
                "Français": "📏 Taille des fiches :",
                "Español": "📏 Tamaño de las tarjetas:",
                "العربية": "📏 حجم البطاقات:",
                "Arabski (libański dialekt)": "📏 حجم البطاقات:",
                "中文": "📏 卡片大小：",
                "日本語": "📏 カードサイズ："
            },
            "Size - Large": {
                "Polski": "Duże (800×600)",
                "English": "Large (800×600)",
                "Deutsch": "Groß (800×600)",
                "Українська": "Великі (800×600)",
                "Français": "Grandes (800×600)",
                "Español": "Grandes (800×600)",
                "العربية": "كبيرة (800×600)",
                "Arabski (libański dialekt)": "كبيرة (800×600)",
                "中文": "大（800×600）",
                "日本語": "大（800×600）"
            },
            "Size - Medium": {
                "Polski": "Średnie (600×450)",
                "English": "Medium (600×450)",
                "Deutsch": "Mittel (600×450)",
                "Українська": "Середні (600×450)",
                "Français": "Moyennes (600×450)",
                "Español": "Medianas (600×450)",
                "العربية": "متوسطة (600×450)",
                "Arabski (libański dialekt)": "متوسطة (600×450)",
                "中文": "中（600×450）",
                "日本語": "中（600×450）"
            },
            "Size - Small": {
                "Polski": "Małe (400×300)",
                "English": "Small (400×300)",
                "Deutsch": "Klein (400×300)",
                "Українська": "Малі (400×300)",
                "Français": "Petites (400×300)",
                "Español": "Pequeñas (400×300)",
                "العربية": "صغيرة (400×300)",
                "Arabski (libański dialekt)": "صغيرة (400×300)",
                "中文": "小（400×300）",
                "日本語": "小（400×300）"
            },
            "Generated flashcards": {
                "Polski": "Wygenerowane fiszki:",
                "English": "Generated flashcards:",
                "Deutsch": "Generierte Karteikarten:",
                "Українська": "Згенеровані картки:",
                "Français": "Fiches générées :",
                "Español": "Tarjetas generadas:",
                "العربية": "البطاقات المُنشأة:",
                "Arabski (libański dialekt)": "الفلاش كاردز اللي نعملت:",
                "中文": "生成的卡片：",
                "日本語": "生成されたフラッシュカード："
            },
            "Download flashcards to print": {
                "Polski": "Pobierz fiszki do wydruku",
                "English": "Download flashcards to print",
                "Deutsch": "Karteikarten zum Drucken herunterladen",
                "Українська": "Завантажити картки для друку",
                "Français": "Télécharger les fiches à imprimer",
                "Español": "Descargar tarjetas para imprimir",
                "العربية": "نزّل بطاقات للطباعة",
                "Arabski (libański dialekt)": "نزّل بطاقات للطباعة",
                "中文": "下载可打印的卡片",
                "日本語": "印刷用カードをダウンロード"
            },
            "Download flashcards": {
                "Polski": "📥 Pobierz fiszki",
                "English": "📥 Download flashcards",
                "Deutsch": "📥 Karteikarten herunterladen",
                "Українська": "📥 Завантажити картки",
                "Français": "📥 Télécharger les fiches",
                "Español": "📥 Descargar tarjetas",
                "العربية": "📥 نزّل البطاقات",
                "Arabski (libański dialekt)": "📥 نزّل البطاقات",
                "中文": "📥 下载卡片",
                "日本語": "📥 カードをダウンロード"
            },
            "Success: pronunciation analyzed": {
                "Polski": "✅ Analiza wymowy gotowa!",
                "English": "✅ Pronunciation analysis ready!",
                "Deutsch": "✅ Ausspracheanalyse fertig!",
                "Українська": "✅ Аналіз вимови готовий!",
                "Français": "✅ Analyse de la prononciation prête !",
                "Español": "✅ ¡Análisis de pronunciación listo!",
                "العربية": "✅ تحليل النطق جاهز!",
                "Arabski (libański dialekt)": "✅ تحليل النطق جاهز!",
                "中文": "✅ 发音分析已完成！",
                "日本語": "✅ 発音分析の準備ができました！"
            },
            "Error: pronunciation not analyzed": {
                "Polski": "❌ Nie udało się przeanalizować wymowy.",
                "English": "❌ Failed to analyze pronunciation.",
                "Deutsch": "❌ Aussprache konnte nicht analysiert werden.",
                "Українська": "❌ Не вдалося проаналізувати вимову.",
                "Français": "❌ Échec de l'analyse de la prononciation.",
                "Español": "❌ No se pudo analizar la pronunciación.",
                "العربية": "❌ فشل تحليل النطق.",
                "Arabski (libański dialekt)": "❌ ما قدرنا نحلّل النطق.",
                "中文": "❌ 无法分析发音。",
                "日本語": "❌ 発音を分析できませんでした。"
            },
            "Error: pronunciation exception": {
                "Polski": "❌ Błąd analizy wymowy:",
                "English": "❌ Pronunciation analysis error:",
                "Deutsch": "❌ Fehler bei der Ausspracheanalyse:",
                "Українська": "❌ Помилка аналізу вимови:",
                "Français": "❌ Erreur d'analyse de la prononciation :",
                "Español": "❌ Error en el análisis de la pronunciación:",
                "العربية": "❌ خطأ في تحليل النطق:",
                "Arabski (libański dialekt)": "❌ خطأ بتحليل النطق:",
                "中文": "❌ 发音分析错误：",
                "日本語": "❌ 発音分析エラー："
            },
            "Warn: enter text to translate": {
                "Polski": "⚠️ Wpisz tekst do przetłumaczenia.",
                "English": "⚠️ Enter text to translate.",
                "Deutsch": "⚠️ Text zum Übersetzen eingeben.",
                "Українська": "⚠️ Введіть текст для перекладу.",
                "Français": "⚠️ Entrez le texte à traduire.",
                "Español": "⚠️ Introduce texto para traducir.",
                "العربية": "⚠️ أدخل نصًا للترجمة.",
                "Arabski (libański dialekt)": "⚠️ اكتب نص للترجمة.",
                "中文": "⚠️ 输入要翻译的文本。",
                "日本語": "⚠️ 翻訳するテキストを入力してください。"
            },
            "Warn: enter text to explain": {
                "Polski": "Wpisz tekst do wyjaśnienia.",
                "English": "Enter text for explanation.",
                "Deutsch": "Text zur Erklärung eingeben.",
                "Українська": "Введіть текст для пояснення.",
                "Français": "Entrez le texte à expliquer.",
                "Español": "Introduce texto para explicar.",
                "العربية": "أدخل نصًا للتوضيح.",
                "Arabski (libański dialekt)": "اكتب نص للتوضيح.",
                "中文": "输入要解释的文本。",
                "日本語": "説明するテキストを入力してください。"
            },
            "Warn: enter text to improve": {
                "Polski": "Wpisz tekst do poprawy stylistycznej.",
                "English": "Enter text to improve style.",
                "Deutsch": "Text zur Stilverbesserung eingeben.",
                "Українська": "Введіть текст для покращення стилю.",
                "Français": "Entrez un texte à améliorer.",
                "Español": "Introduce texto para mejorar el estilo.",
                "العربية": "أدخل نصًا لتحسين الأسلوب.",
                "Arabski (libański dialekt)": "اكتب نص للتجميل.",
                "中文": "输入要改进风格的文本。",
                "日本語": "スタイルを改善するテキストを入力してください。"
            },
            "Warn: enter text to generate flashcards": {
                "Polski": "Wpisz tekst do wygenerowania fiszek.",
                "English": "Enter text to generate flashcards.",
                "Deutsch": "Text eingeben, um Karteikarten zu erstellen.",
                "Українська": "Введіть текст для створення карток.",
                "Français": "Entrez un texte pour générer des fiches.",
                "Español": "Introduce texto para generar tarjetas.",
                "العربية": "أدخل نصًا لإنشاء بطاقات.",
                "Arabski (libański dialekt)": "اكتب نص لتوليد بطاقات.",
                "中文": "输入文本以生成卡片。",
                "日本語": "フラッシュカードを生成するテキストを入力してください。"
            },
            "Result": {
                "Polski": "Wynik",
                "English": "Result",
                "Deutsch": "Ergebnis",
                "Українська": "Результат",
                "Français": "Résultat",
                "Español": "Resultado",
                "العربية": "النتيجة",
                "Arabski (libański dialekt)": "النتيجة",
                "中文": "结果",
                "日本語": "結果"
            },
            "Translation": {
                "Polski": "Tłumaczenie:",
                "English": "Translation:",
                "Deutsch": "Übersetzung:",
                "Українська": "Переклад:",
                "Français": "Traduction :",
                "Español": "Traducción:",
                "العربية": "الترجمة:",
                "Arabski (libański dialekt)": "الترجمة:",
                "中文": "翻译：",
                "日本語": "翻訳："
            },
            "Listen translation": {
                "Polski": "🔊 Odsłuchaj tłumaczenie",
                "English": "🔊 Listen to translation",
                "Deutsch": "🔊 Übersetzung anhören",
                "Українська": "🔊 Прослухати переклад",
                "Français": "🔊 Écouter la traduction",
                "Español": "🔊 Escuchar la traducción",
                "العربية": "🔊 استمع إلى الترجمة",
                "Arabski (libański dialekt)": "🔊 اسمع الترجمة",
                "中文": "🔊 听翻译",
                "日本語": "🔊 翻訳を聴く"
            },
            "Pronunciation analysis": {
                "Polski": "📊 Analiza wymowy:",
                "English": "📊 Pronunciation analysis:",
                "Deutsch": "📊 Ausspracheanalyse:",
                "Українська": "📊 Аналіз вимови:",
                "Français": "📊 Analyse de la prononciation :",
                "Español": "📊 Análisis de pronunciación:",
                "العربية": "📊 تحليل النطق:",
                "Arabski (libański dialekt)": "📊 تحليل النطق:",
                "中文": "📊 发音分析：",
                "日本語": "📊 発音分析："
            },
            "From cache": {
                "Polski": "📋 Wynik z cache",
                "English": "📋 Result from cache",
                "Deutsch": "📋 Ergebnis aus dem Cache",
                "Українська": "📋 Результат з кешу",
                "Français": "📋 Résultat depuis le cache",
                "Español": "📋 Resultado desde caché",
                "العربية": "📋 نتيجة من الذاكرة المؤقتة",
                "Arabski (libański dialekt)": "📋 نتيجة من الكاش",
                "中文": "📋 缓存结果",
                "日本語": "📋 キャッシュからの結果"
            },
            "Flashcards image title": {
                "Polski": "📚 FISZKI DO NAUKI",
                "English": "📚 FLASHCARDS FOR LEARNING",
                "Deutsch": "📚 LERN-KARTEIKARTEN",
                "Українська": "📚 КАРТКИ ДЛЯ НАВЧАННЯ",
                "Français": "📚 FICHES D'APPRENTISSAGE",
                "Español": "📚 TARJETAS PARA APRENDER",
                "العربية": "📚 بطاقات للتعلم",
                "Arabski (libański dialekt)": "📚 بطاقات للتعلّم",
                "中文": "📚 学习卡片",
                "日本語": "📚 学習用フラッシュカード"
            },
            "Flashcard label - word": {
                "Polski": "SŁÓWKO:",
                "English": "WORD:",
                "Deutsch": "WORT:",
                "Українська": "СЛОВО:",
                "Français": "MOT :",
                "Español": "PALABRA:",
                "العربية": "الكلمة:",
                "Arabski (libański dialekt)": "الكلمة:",
                "中文": "词语：",
                "日本語": "単語："
            },
            "Flashcard label - definition": {
                "Polski": "DEFINICJA:",
                "English": "DEFINITION:",
                "Deutsch": "DEFINITION:",
                "Українська": "ВИЗНАЧЕННЯ:",
                "Français": "DÉFINITION :",
                "Español": "DEFINICIÓN:",
                "العربية": "التعريف:",
                "Arabski (libański dialekt)": "التعريف:",
                "中文": "定义：",
                "日本語": "定義："
            },
            "Flashcard label - example": {
                "Polski": "PRZYKŁAD:",
                "English": "EXAMPLE:",
                "Deutsch": "BEISPIEL:",
                "Українська": "ПРИКЛАД:",
                "Français": "EXEMPLE :",
                "Español": "EJEMPLO:",
                "العربية": "مثال:",
                "Arabski (libański dialekt)": "مثال:",
                "中文": "例子：",
                "日本語": "例："
            },
            "Success: mic recognized": {
                "Polski": "✅ Nagrano i rozpoznano! Tekst dodano powyżej.",
                "English": "✅ Recorded and recognized! Text added above.",
                "Deutsch": "✅ Aufgenommen und erkannt! Text oben hinzugefügt.",
                "Українська": "✅ Записано і розпізнано! Текст додано вище.",
                "Français": "✅ Enregistré et reconnu ! Texte ajouté ci-dessus.",
                "Español": "✅ Grabado y reconocido. Texto añadido arriba.",
                "العربية": "✅ تم التسجيل والتعرف! تمت إضافة النص أعلاه.",
                "Arabski (libański dialekt)": "✅ تسجّل وتعرّف! نضاف النص فوق.",
                "中文": "✅ 已录制并识别！文本已添加在上方。",
                "日本語": "✅ 録音して認識しました。上にテキストを追加しました。"
            },
            "Warn: mic not recognized": {
                "Polski": "⚠️ Nie udało się rozpoznać mowy.",
                "English": "⚠️ Could not recognize speech.",
                "Deutsch": "⚠️ Sprache konnte nicht erkannt werden.",
                "Українська": "⚠️ Не вдалося розпізнати мовлення.",
                "Français": "⚠️ Impossible de reconnaître la parole.",
                "Español": "⚠️ No se pudo reconocer el habla.",
                "العربية": "⚠️ تعذّر التعرّف على الكلام.",
                "Arabski (libański dialekt)": "⚠️ ما قدر يتعرّف على الحكي.",
                "中文": "⚠️ 语音无法识别。",
                "日本語": "⚠️ 音声を認識できませんでした。"
            },
            "Success: file recognized": {
                "Polski": "✅ Wczytano i rozpoznano! Tekst dodano powyżej.",
                "English": "✅ Loaded and recognized! Text added above.",
                "Deutsch": "✅ Geladen und erkannt! Text oben hinzugefügt.",
                "Українська": "✅ Завантажено і розпізнано! Текст додано вище.",
                "Français": "✅ Chargé et reconnu ! Texte ajouté ci-dessus.",
                "Español": "✅ Cargado y reconocido. Texto añadido arriba.",
                "العربية": "✅ تم التحميل والتعرف! تمت إضافة النص أعلاه.",
                "Arabski (libański dialekt)": "✅ نزل وتعرّف! نضاف النص فوق.",
                "中文": "✅ 已加载并识别！文本已添加在上方。",
                "日本語": "✅ 読み込み、認識しました。上にテキストを追加しました。"
            },
            "Warn: file not recognized": {
                "Polski": "⚠️ Nie udało się rozpoznać mowy z pliku.",
                "English": "⚠️ Could not recognize speech from file.",
                "Deutsch": "⚠️ Sprache aus der Datei konnte nicht erkannt werden.",
                "Українська": "⚠️ Не вдалося розпізнати мовлення з файлу.",
                "Français": "⚠️ Impossible de reconnaître la parole à partir du fichier.",
                "Español": "⚠️ No se pudo reconocer el habla desde el archivo.",
                "العربية": "⚠️ تعذّر التعرّف على الكلام من الملف.",
                "Arabski (libański dialekt)": "⚠️ ما قدر يتعرّف على الحكي من الملف.",
                "中文": "⚠️ 无法从文件中识别语音。",
                "日本語": "⚠️ ファイルから音声を認識できませんでした。"
            },
            "Success: words generated": {
                "Polski": "✅ Wygenerowano słowa do ćwiczenia!",
                "English": "✅ Words for practice generated!",
                "Deutsch": "✅ Wörter zum Üben generiert!",
                "Українська": "✅ Згенеровано слова для практики!",
                "Français": "✅ Mots pour la pratique générés !",
                "Español": "✅ Palabras para practicar generadas!",
                "العربية": "✅ تم توليد كلمات للتمرين!",
                "Arabski (libański dialekt)": "✅ تولّدوا كلمات للتمرين!",
                "中文": "✅ 已生成练习单词！",
                "日本語": "✅ 練習用の単語を生成しました！"
            },
            "Error: words not generated": {
                "Polski": "❌ Nie udało się wygenerować słów do ćwiczenia.",
                "English": "❌ Failed to generate words for practice.",
                "Deutsch": "❌ Wörter zum Üben konnten nicht generiert werden.",
                "Українська": "❌ Не вдалося згенерувати слова для практики.",
                "Français": "❌ Échec de génération des mots pour la pratique.",
                "Español": "❌ No se pudieron generar palabras para practicar.",
                "العربية": "❌ فشل توليد كلمات للتمرين.",
                "Arabski (libański dialekt)": "❌ ما قدرنا نولّد كلمات للتمرين.",
                "中文": "❌ 生成练习单词失败。",
                "日本語": "❌ 練習用の単語を生成できませんでした。"
            },
            "Error: words generation exception": {
                "Polski": "❌ Błąd generowania słów:",
                "English": "❌ Error generating words:",
                "Deutsch": "❌ Fehler beim Generieren von Wörtern:",
                "Українська": "❌ Помилка генерації слів:",
                "Français": "❌ Erreur lors de la génération des mots :",
                "Español": "❌ Error al generar palabras:",
                "العربية": "❌ خطأ في توليد الكلمات:",
                "Arabski (libański dialekt)": "❌ خطأ بتوليد الكلمات:",
                "中文": "❌ 生成单词时出错：",
                "日本語": "❌ 単語生成エラー："
            },
            "Clear practice result": {
                "Polski": "🧹 Wyczyść wynik ćwiczeń",
                "English": "🧹 Clear practice result",
                "Deutsch": "🧹 Übungsergebnis löschen",
                "Українська": "🧹 Очистити результат вправи",
                "Français": "🧹 Effacer le résultat d'entraînement",
                "Español": "🧹 Limpiar resultado de práctica",
                "العربية": "🧹 مسح نتيجة التمرين",
                "Arabski (libański dialekt)": "🧹 امسح نتيجة التمرين",
                "中文": "🧹 清除练习结果",
                "日本語": "🧹 練習結果をクリア"
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
            },
            # Fiszki – wybór języka definicji
            "Wybierz język definicji fiszek": {
                "Polski": "Wybierz język definicji fiszek",
                "English": "Choose flashcard definition language",
                "Deutsch": "Wähle die Sprache der Definitionen",
                "Українська": "Виберіть мову визначень для карток",
                "Français": "Choisissez la langue des définitions",
                "Español": "Elige el idioma de las definiciones",
                "العربية": "اختر لغة التعاريف",
                "Arabski (libański dialekt)": "اختر لغة التعاريف (لبناني)",
                "中文": "选择定义语言",
                "日本語": "定義の言語を選択"
            },
            "Język definicji": {
                "Polski": "Język definicji",
                "English": "Definition language",
                "Deutsch": "Definitionssprache",
                "Українська": "Мова визначень",
                "Français": "Langue des définitions",
                "Español": "Idioma de las definiciones",
                "العربية": "لغة التعاريف",
                "Arabski (libański dialekt)": "لغة التعاريف (لبناني)",
                "中文": "定义语言",
                "日本語": "定義の言語"
            },
            "Język interfejsu (opcja)": {
                "Polski": "Język interfejsu",
                "English": "Interface language",
                "Deutsch": "Interface-Sprache",
                "Українська": "Мова інтерфейсу",
                "Français": "Langue de l'interface",
                "Español": "Idioma de la interfaz",
                "العربية": "لغة الواجهة",
                "Arabski (libański dialekt)": "لغة الواجهة (لبناني)",
                "中文": "界面语言",
                "日本語": "インターフェースの言語"
            }
            ,
            "Ustawienia": {
                "Polski": "⚙️ Ustawienia",
                "English": "⚙️ Settings",
                "Deutsch": "⚙️ Einstellungen",
                "Українська": "⚙️ Налаштування",
                "Français": "⚙️ Paramètres",
                "Español": "⚙️ Configuración",
                "العربية": "⚙️ الإعدادات",
                "Arabski (libański dialekt)": "⚙️ الإعدادات",
                "中文": "⚙️ 设置",
                "日本語": "⚙️ 設定"
            },
            "Język interfejsu": {
                "Polski": "🌐 Język interfejsu",
                "English": "🌐 Interface language",
                "Deutsch": "🌐 Interface-Sprache",
                "Українська": "🌐 Мова інтерфейсу",
                "Français": "🌐 Langue de l'interface",
                "Español": "🌐 Idioma de la interfaz",
                "العربية": "🌐 لغة الواجهة",
                "Arabski (libański dialekt)": "🌐 لغة الواجهة (لبناني)",
                "中文": "🌐 界面语言",
                "日本語": "🌐 インターフェースの言語"
            },
            "Motyw": {
                "Polski": "🎨 Motyw",
                "English": "🎨 Theme",
                "Deutsch": "🎨 Thema",
                "Українська": "🎨 Тема",
                "Français": "🎨 Thème",
                "Español": "🎨 Tema",
                "العربية": "🎨 السمة",
                "Arabski (libański dialekt)": "🎨 السمة",
                "中文": "🎨 主题",
                "日本語": "🎨 テーマ"
            },
            "Kolor tła": {
                "Polski": "Kolor tła",
                "English": "Background color",
                "Deutsch": "Hintergrundfarbe",
                "Українська": "Колір тла",
                "Français": "Couleur d'arrière-plan",
                "Español": "Color de fondo",
                "العربية": "لون الخلفية",
                "Arabski (libański dialekt)": "لون الخلفية",
                "中文": "背景颜色",
                "日本語": "背景色"
            },
            "Jasny": {
                "Polski": "Jasny",
                "English": "Light",
                "Deutsch": "Hell",
                "Українська": "Світлий",
                "Français": "Clair",
                "Español": "Claro",
                "العربية": "فاتح",
                "Arabski (libański dialekt)": "فاتح",
                "中文": "浅色",
                "日本語": "ライト"
            },
            "Ciemny": {
                "Polski": "Ciemny",
                "English": "Dark",
                "Deutsch": "Dunkel",
                "Українська": "Темний",
                "Français": "Sombre",
                "Español": "Oscuro",
                "العربية": "داكن",
                "Arabski (libański dialekt)": "داكن",
                "中文": "深色",
                "日本語": "ダーク"
            },
            "O aplikacji": {
                "Polski": "ℹ️ O aplikacji",
                "English": "ℹ️ About the app",
                "Deutsch": "ℹ️ Über die App",
                "Українська": "ℹ️ Про застосунок",
                "Français": "ℹ️ À propos de l'app",
                "Español": "ℹ️ Acerca de la app",
                "العربية": "ℹ️ حول التطبيق",
                "Arabski (libański dialekt)": "ℹ️ عن التطبيق",
                "中文": "ℹ️ 关于应用",
                "日本語": "ℹ️ アプリについて"
            },
            "Ćwicz wymowę": {
                "Polski": "🎤 Ćwicz wymowę",
                "English": "🎤 Practice pronunciation",
                "Deutsch": "🎤 Aussprache üben",
                "Українська": "🎤 Тренуйте вимову",
                "Français": "🎤 Exercer la prononciation",
                "Español": "🎤 Practicar la pronunciación",
                "العربية": "🎤 تدرب على النطق",
                "Arabski (libański dialekt)": "🎤 تمرن على النطق",
                "中文": "🎤 练习发音",
                "日本語": "🎤 発音練習"
            },
            "Język do ćwiczenia": {
                "Polski": "🌍 Język do ćwiczenia",
                "English": "🌍 Language to practice",
                "Deutsch": "🌍 Übungssprache",
                "Українська": "🌍 Мова для практики",
                "Français": "🌍 Langue à pratiquer",
                "Español": "🌍 Idioma para practicar",
                "العربية": "🌍 اللغة للممارسة",
                "Arabski (libański dialekt)": "🌍 اللغة للتدريب",
                "中文": "🌍 练习语言",
                "日本語": "🌍 練習する言語"
            },
            "Typ ćwiczenia": {
                "Polski": "🎯 Typ ćwiczenia",
                "English": "🎯 Exercise type",
                "Deutsch": "🎯 Übungstyp",
                "Українська": "🎯 Тип вправи",
                "Français": "🎯 Type d'exercice",
                "Español": "🎯 Tipo de ejercicio",
                "العربية": "🎯 نوع التمرين",
                "Arabski (libański dialekt)": "🎯 نوع التمرين",
                "中文": "🎯 练习类型",
                "日本語": "🎯 練習の種類"
            },
            "Generuj słowa do ćwiczenia": {
                "Polski": "🎲 Generuj słowa do ćwiczenia",
                "English": "🎲 Generate words to practice",
                "Deutsch": "🎲 Wörter zum Üben generieren",
                "Українська": "🎲 Згенерувати слова для практики",
                "Français": "🎲 Générer des mots à pratiquer",
                "Español": "🎲 Generar palabras para practicar",
                "العربية": "🎲 أنشئ كلمات للتمرن",
                "Arabski (libański dialekt)": "🎲 ولّد كلمات للتدريب",
                "中文": "🎲 生成练习单词",
                "日本語": "🎲 練習用の単語を生成"
            },
            "Nagraj wymowę": {
                "Polski": "🎤 Nagraj wymowę",
                "English": "🎤 Record pronunciation",
                "Deutsch": "🎤 Aussprache aufnehmen",
                "Українська": "🎤 Запишіть вимову",
                "Français": "🎤 Enregistrer la prononciation",
                "Español": "🎤 Grabar pronunciación",
                "العربية": "🎤 سجّل النطق",
                "Arabski (libański dialekt)": "🎤 سجّل النطق",
                "中文": "🎤 录制发音",
                "日本語": "🎤 発音を録音"
            },
            "Rozpoznano wymowę": {
                "Polski": "✅ Rozpoznano wymowę",
                "English": "✅ Pronunciation recognized",
                "Deutsch": "✅ Aussprache erkannt",
                "Українська": "✅ Вимову розпізнано",
                "Français": "✅ Prononciation reconnue",
                "Español": "✅ Pronunciación reconocida",
                "العربية": "✅ تم التعرف على النطق",
                "Arabski (libański dialekt)": "✅ تم التعرف على النطق",
                "中文": "✅ 已识别发音",
                "日本語": "✅ 発音が認識されました"
            },
            "Ostatnia rozpoznana wypowiedź:": {
                "Polski": "🔎 Ostatnia rozpoznana wypowiedź:",
                "English": "🔎 Last recognized utterance:",
                "Deutsch": "🔎 Zuletzt erkannte Äußerung:",
                "Українська": "🔎 Останнє розпізнане висловлювання:",
                "Français": "🔎 Dernière énonciation reconnue :",
                "Español": "🔎 Última intervención reconocida:",
                "العربية": "🔎 آخر جملة تم التعرف عليها:",
                "Arabski (libański dialekt)": "🔎 آخر جملة تم التعرف عليها:",
                "中文": "🔎 最近识别的话语：",
                "日本語": "🔎 最後に認識された発話："
            },
            "Analizuj wymowę": {
                "Polski": "🔍 Analizuj wymowę",
                "English": "🔍 Analyze pronunciation",
                "Deutsch": "🔍 Aussprache analysieren",
                "Українська": "🔍 Проаналізувати вимову",
                "Français": "🔍 Analyser la prononciation",
                "Español": "🔍 Analizar pronunciación",
                "العربية": "🔍 حلل النطق",
                "Arabski (libański dialekt)": "🔍 حلل النطق",
                "中文": "🔍 分析发音",
                "日本語": "🔍 発音を分析"
            },
            "Liczba requestów": {
                "Polski": "📊 Liczba requestów",
                "English": "📊 Number of requests",
                "Deutsch": "📊 Anzahl der Anfragen",
                "Українська": "📊 Кількість запитів",
                "Français": "📊 Nombre de requêtes",
                "Español": "📊 Número de solicitudes",
                "العربية": "📊 عدد الطلبات",
                "Arabski (libański dialekt)": "📊 عدد الطلبات",
                "中文": "📊 请求数量",
                "日本語": "📊 リクエスト数"
            },
            "Popraw błędy przed tłumaczeniem": {
                "Polski": "🔧 Popraw błędy przed tłumaczeniem",
                "English": "🔧 Correct errors before translating",
                "Deutsch": "🔧 Fehler vor der Übersetzung korrigieren",
                "Українська": "🔧 Виправити помилки перед перекладом",
                "Français": "🔧 Corriger les erreurs avant la traduction",
                "Español": "🔧 Corregir errores antes de traducir",
                "العربية": "🔧 صحّح الأخطاء قبل الترجمة",
                "Arabski (libański dialekt)": "🔧 صحّح الأخطاء قبل الترجمة",
                "中文": "🔧 翻译前纠正错误",
                "日本語": "🔧 翻訳前にエラーを修正"
            },
            "Help: Popraw błędy przed tłumaczeniem": {
                "Polski": "Popraw błędy gramatyczne i stylistyczne w oryginalnym języku przed tłumaczeniem",
                "English": "Correct grammar and style in the original language before translating",
                "Deutsch": "Korrigiere Grammatik und Stil in der Ausgangssprache vor der Übersetzung",
                "Українська": "Виправити граматику і стиль в оригінальній мові перед перекладом",
                "Français": "Corriger la grammaire et le style dans la langue d'origine avant la traduction",
                "Español": "Corregir gramática y estilo en el idioma original antes de traducir",
                "العربية": "صحّح القواعد والأسلوب في اللغة الأصلية قبل الترجمة",
                "Arabski (libański dialekt)": "صحّح القواعد والأسلوب في اللغة الأصلية قبل الترجمة",
                "中文": "在翻译之前先在原始语言中纠正语法和风格",
                "日本語": "翻訳前に原文の文法とスタイルを修正する"
            },
            "Placeholder: tłumaczenie": {
                "Polski": "Wpisz tutaj tekst do przetłumaczenia...",
                "English": "Enter text here to translate...",
                "Deutsch": "Text hier zum Übersetzen eingeben...",
                "Українська": "Введіть тут текст для перекладу...",
                "Français": "Saisissez ici le texte à traduire...",
                "Español": "Introduce aquí el texto a traducir...",
                "العربية": "أدخل هنا النص للترجمة...",
                "Arabski (libański dialekt)": "اكتب هون النص للترجمة...",
                "中文": "在此输入要翻译的文本...",
                "日本語": "ここに翻訳するテキストを入力..."
            },
            "Placeholder: wyjaśnienia": {
                "Polski": "Wpisz tutaj tekst do wyjaśnienia...",
                "English": "Enter text here for explanation...",
                "Deutsch": "Text hier zur Erklärung eingeben...",
                "Українська": "Введіть тут текст для пояснення...",
                "Français": "Saisissez ici le texte à expliquer...",
                "Español": "Introduce aquí el texto para explicar...",
                "العربية": "أدخل هنا نصاً لشرحه...",
                "Arabski (libański dialekt)": "اكتب هون نص للتوضيح...",
                "中文": "在此输入要解释的文本...",
                "日本語": "ここに説明するテキストを入力..."
            },
            "Placeholder: stylistyka": {
                "Polski": "Wpisz tutaj tekst do poprawy...",
                "English": "Enter text here to improve...",
                "Deutsch": "Text hier zur Verbesserung eingeben...",
                "Українська": "Введіть тут текст для покращення...",
                "Français": "Saisissez ici le texte à améliorer...",
                "Español": "Introduce aquí el texto a mejorar...",
                "العربية": "أدخل هنا نصاً لتحسينه...",
                "Arabski (libański dialekt)": "اكتب هون نص للتجميل...",
                "中文": "在此输入要改进的文本...",
                "日本語": "ここに改善するテキストを入力..."
            },
            "Placeholder: fiszki": {
                "Polski": "Wpisz tutaj tekst do wygenerowania fiszek...",
                "English": "Enter text here to generate flashcards...",
                "Deutsch": "Text hier eingeben, um Karteikarten zu erstellen...",
                "Українська": "Введіть тут текст для створення карток...",
                "Français": "Saisissez ici le texte pour générer des fiches...",
                "Español": "Introduce aquí el texto para generar tarjetas...",
                "العربية": "أدخل هنا نصاً لإنشاء بطاقات...",
                "Arabski (libański dialekt)": "اكتب هون نص لتوليد بطاقات...",
                "中文": "在此输入文本以生成卡片...",
                "日本語": "ここにカードを生成するテキストを入力..."
            },
            "Opt - Słowa podstawowe": {
                "Polski": "Słowa podstawowe",
                "English": "Basic words",
                "Deutsch": "Grundwörter",
                "Українська": "Базові слова",
                "Français": "Mots de base",
                "Español": "Palabras básicas",
                "العربية": "كلمات أساسية",
                "Arabski (libański dialekt)": "كلمات أساسية",
                "中文": "基础词汇",
                "日本語": "基本単語"
            },
            "Opt - Zwroty codzienne": {
                "Polski": "Zwroty codzienne",
                "English": "Daily phrases",
                "Deutsch": "Alltägliche Redewendungen",
                "Українська": "Повсякденні фрази",
                "Français": "Phrases quotidiennes",
                "Español": "Frases cotidianas",
                "العربية": "عبارات يومية",
                "Arabski (libański dialekt)": "عبارات يومية",
                "中文": "日常用语",
                "日本語": "日常フレーズ"
            },
            "Opt - Liczby": {
                "Polski": "Liczby",
                "English": "Numbers",
                "Deutsch": "Zahlen",
                "Українська": "Числа",
                "Français": "Nombres",
                "Español": "Números",
                "العربية": "الأرقام",
                "Arabski (libański dialekt)": "الأرقام",
                "中文": "数字",
                "日本語": "数字"
            },
            "Opt - Kolory": {
                "Polski": "Kolory",
                "English": "Colors",
                "Deutsch": "Farben",
                "Українська": "Кольори",
                "Français": "Couleurs",
                "Español": "Colores",
                "العربية": "الألوان",
                "Arabski (libański dialekt)": "الألوان",
                "中文": "颜色",
                "日本語": "色"
            },
            "Opt - Członkowie rodziny": {
                "Polski": "Członkowie rodziny",
                "English": "Family members",
                "Deutsch": "Familienmitglieder",
                "Українська": "Члени родини",
                "Français": "Membres de la famille",
                "Español": "Miembros de la familia",
                "العربية": "أفراد العائلة",
                "Arabski (libański dialekt)": "أفراد العيلة",
                "中文": "家庭成员",
                "日本語": "家族"
            },
            "About content": {
                "Polski": """
        **Tłumacz Wielojęzyczny** to zaawansowane narzędzie do:
        - 🌍 Tłumaczenia tekstów
        - 📚 Wyjaśniania gramatyki
        - ✨ Poprawy stylistyki
        - 🔧 Korekcji błędów
        - 📖 Tworzenia fiszek
        - 🎤 Ćwiczenia wymowy
        """,
                "English": """
        **Multilingual Translator** helps you:
        - 🌍 Translate texts
        - 📚 Explain vocabulary and grammar
        - ✨ Improve style (polish your text)
        - 🔧 Correct errors
        - 📖 Create flashcards
        - 🎤 Practice pronunciation
        """,
                "Deutsch": """
        **Mehrsprachiger Übersetzer** – Funktionen:
        - 🌍 Texte übersetzen
        - 📚 Wortschatz und Grammatik erklären
        - ✨ Stil verbessern
        - 🔧 Fehler korrigieren
        - 📖 Karteikarten erstellen
        - 🎤 Aussprache üben
        """,
                "Українська": """
        **Багатомовний перекладач** допомагає:
        - 🌍 Перекладати тексти
        - 📚 Пояснювати лексику та граматику
        - ✨ Покращувати стиль
        - 🔧 Виправляти помилки
        - 📖 Створювати картки
        - 🎤 Тренувати вимову
        """,
                "Français": """
        **Traducteur multilingue** permet de :
        - 🌍 Traduire des textes
        - 📚 Expliquer vocabulaire et grammaire
        - ✨ Améliorer le style
        - 🔧 Corriger les erreurs
        - 📖 Créer des fiches
        - 🎤 S'entraîner à la prononciation
        """,
                "Español": """
        **Traductor multilingüe** te ayuda a:
        - 🌍 Traducir textos
        - 📚 Explicar vocabulario y gramática
        - ✨ Mejorar el estilo
        - 🔧 Corregir errores
        - 📖 Crear tarjetas
        - 🎤 Practicar la pronunciación
        """,
                "العربية": """
        **مترجم متعدد اللغات** يساعدك على:
        - 🌍 ترجمة النصوص
        - 📚 شرح المفردات والقواعد
        - ✨ تحسين الأسلوب
        - 🔧 تصحيح الأخطاء
        - 📖 إنشاء بطاقات تعليمية
        - 🎤 التدرب على النطق
        """,
                "Arabski (libański dialekt)": """
        **مترجم متعدد اللغات** بيساعدك:
        - 🌍 تترجم نصوص
        - 📚 تشرح كلمات وقواعد
        - ✨ تحسن الأسلوب
        - 🔧 تصحّح أخطاء
        - 📖 تعمل فلاش كاردز
        - 🎤 تتمرن على النطق
        """,
                "中文": """
        **多语言翻译器** 帮助你：
        - 🌍 翻译文本
        - 📚 解释词汇和语法
        - ✨ 改进文风
        - 🔧 纠正错误
        - 📖 创建学习卡片
        - 🎤 练习发音
        """,
                "日本語": """
        **多言語翻訳ツール** は次のことができます：
        - 🌍 テキストの翻訳
        - 📚 語彙と文法の説明
        - ✨ スタイルの改善
        - 🔧 誤りの修正
        - 📖 フラッシュカードの作成
        - 🎤 発音練習
        """,
            },
            "Style caption": {
                "Polski": "Nie tłumaczy — tylko poprawa stylu i gramatyki w tym samym języku.",
                "English": "No translation — improves style and grammar in the same language.",
                "Deutsch": "Keine Übersetzung — verbessert Stil und Grammatik in derselben Sprache.",
                "Українська": "Без перекладу — лише покращення стилю та граматики тією ж мовою.",
                "Français": "Pas de traduction — améliore le style et la grammaire dans la même langue.",
                "Español": "Sin traducción: mejora el estilo y la gramática en el mismo idioma.",
                "العربية": "لا ترجمة — تحسين الأسلوب والقواعد باللغة نفسها.",
                "Arabski (libański dialekt)": "ما في ترجمة — بس تحسين الأسلوب والقواعد بنفس اللغة.",
                "中文": "不进行翻译——仅在同一语言中改进风格和语法。",
                "日本語": "翻訳はしません。同じ言語で文体と文法のみ改善します。"
            },
            "API stats": {
                "Polski": "📊 Statystyki użycia API",
                "English": "📊 API usage stats",
                "Deutsch": "📊 API-Nutzungsstatistiken",
                "Українська": "📊 Статистика використання API",
                "Français": "📊 Statistiques d'utilisation de l'API",
                "Español": "📊 Estadísticas de uso de la API",
                "العربية": "📊 إحصائيات استخدام API",
                "Arabski (libański dialekt)": "📊 إحصائيات استخدام API",
                "中文": "📊 API 使用统计",
                "日本語": "📊 API 使用統計"
            },
            "Total tokens": {
                "Polski": "🔢 Łącznie tokenów",
                "English": "🔢 Total tokens",
                "Deutsch": "🔢 Gesamtanzahl Tokens",
                "Українська": "🔢 Всього токенів",
                "Français": "🔢 Total de jetons",
                "Español": "🔢 Tokens totales",
                "العربية": "🔢 إجمالي التوكنات",
                "Arabski (libański dialekt)": "🔢 إجمالي التوكنات",
                "中文": "🔢 令牌总数",
                "日本語": "🔢 トークン合計"
            },
            "Total cost": {
                "Polski": "💰 Łączny koszt",
                "English": "💰 Total cost",
                "Deutsch": "💰 Gesamtkosten",
                "Українська": "💰 Загальна вартість",
                "Français": "💰 Coût total",
                "Español": "💰 Costo total",
                "العربية": "💰 التكلفة الإجمالية",
                "Arabski (libański dialekt)": "💰 التكلفة الإجمالية",
                "中文": "💰 总成本",
                "日本語": "💰 総コスト"
            },
            "Last usage": {
                "Polski": "📈 Ostatnie użycie:",
                "English": "📈 Last usage:",
                "Deutsch": "📈 Letzte Nutzung:",
                "Українська": "📈 Останнє використання:",
                "Français": "📈 Dernière utilisation :",
                "Español": "📈 Último uso:",
                "العربية": "📈 آخر استخدام:",
                "Arabski (libański dialekt)": "📈 آخر استخدام:",
                "中文": "📈 最近使用：",
                "日本語": "📈 直近の利用："
            },
            "Model label": {
                "Polski": "Model:",
                "English": "Model:",
                "Deutsch": "Modell:",
                "Українська": "Модель:",
                "Français": "Modèle :",
                "Español": "Modelo:",
                "العربية": "النموذج:",
                "Arabski (libański dialekt)": "الموديل:",
                "中文": "模型：",
                "日本語": "モデル："
            },
            "Input tokens": {
                "Polski": "Tokeny wejściowe:",
                "English": "Input tokens:",
                "Deutsch": "Eingabe-Tokens:",
                "Українська": "Вхідні токени:",
                "Français": "Jetons d'entrée :",
                "Español": "Tokens de entrada:",
                "العربية": "التوكنات الداخلة:",
                "Arabski (libański dialekt)": "التوكنات الداخلة:",
                "中文": "输入令牌：",
                "日本語": "入力トークン："
            },
            "Output tokens": {
                "Polski": "Tokeny wyjściowe:",
                "English": "Output tokens:",
                "Deutsch": "Ausgabe-Tokens:",
                "Українська": "Вихідні токени:",
                "Français": "Jetons de sortie :",
                "Español": "Tokens de salida:",
                "العربية": "التوكنات الخارجة:",
                "Arabski (libański dialekt)": "التوكنات الخارجة:",
                "中文": "输出令牌：",
                "日本語": "出力トークン："
            },
            "Cost label": {
                "Polski": "Koszt:",
                "English": "Cost:",
                "Deutsch": "Kosten:",
                "Українська": "Вартість:",
                "Français": "Coût :",
                "Español": "Costo:",
                "العربية": "التكلفة:",
                "Arabski (libański dialekt)": "الكلفة:",
                "中文": "成本：",
                "日本語": "コスト："
            },
            "Cost history": {
                "Polski": "📊 Historia kosztów",
                "English": "📊 Cost history",
                "Deutsch": "📊 Kostenverlauf",
                "Українська": "📊 Історія витрат",
                "Français": "📊 Historique des coûts",
                "Español": "📊 Historial de costos",
                "العربية": "📊 سجلّ التكلفة",
                "Arabski (libański dialekt)": "📊 سجلّ التكلفة",
                "中文": "📊 成本历史",
                "日本語": "📊 コスト履歴"
            },
            "Reset stats": {
                "Polski": "🔄 Resetuj statystyki",
                "English": "🔄 Reset stats",
                "Deutsch": "🔄 Statistiken zurücksetzen",
                "Українська": "🔄 Скинути статистику",
                "Français": "🔄 Réinitialiser les statistiques",
                "Español": "🔄 Restablecer estadísticas",
                "العربية": "🔄 إعادة تعيين الإحصائيات",
                "Arabski (libański dialekt)": "🔄 صفّر الإحصائيات",
                "中文": "🔄 重置统计",
                "日本語": "🔄 統計をリセット"
            },
            "Footer tagline": {
                "Polski": "🌍 <strong>Tłumacz Wielojęzyczny</strong> - Twoje narzędzie do nauki języków",
                "English": "🌍 <strong>Multilingual Translator</strong> - Your language learning tool",
                "Deutsch": "🌍 <strong>Mehrsprachiger Übersetzer</strong> – Dein Sprachlerntool",
                "Українська": "🌍 <strong>Багатомовний перекладач</strong> – Твій інструмент для вивчення мов",
                "Français": "🌍 <strong>Traducteur multilingue</strong> – Votre outil d'apprentissage des langues",
                "Español": "🌍 <strong>Traductor multilingüe</strong> – Tu herramienta para aprender idiomas",
                "العربية": "🌍 <strong>مترجم متعدد اللغات</strong> – أداتك لتعلم اللغات",
                "Arabski (libański dialekt)": "🌍 <strong>مترجم متعدد اللغات</strong> – أداتك لتعلّم اللغات",
                "中文": "🌍 <strong>多语言翻译器</strong> — 你的语言学习工具",
                "日本語": "🌍 <strong>多言語翻訳ツール</strong> — あなたの語学学習ツール"
            },
            "Footer made with": {
                "Polski": "Made with ❤️ using Streamlit & OpenAI",
                "English": "Made with ❤️ using Streamlit & OpenAI",
                "Deutsch": "Mit ❤️ erstellt mit Streamlit & OpenAI",
                "Українська": "Зроблено з ❤️ на Streamlit і OpenAI",
                "Français": "Fait avec ❤️ grâce à Streamlit & OpenAI",
                "Español": "Hecho con ❤️ usando Streamlit y OpenAI",
                "العربية": "صُنع بحب ❤️ باستخدام Streamlit و OpenAI",
                "Arabski (libański dialekt)": "معمول بمحبة ❤️ باستعمال Streamlit و OpenAI",
                "中文": "用 ❤️ 使用 Streamlit 和 OpenAI 制作",
                "日本語": "Streamlit と OpenAI で ❤️ を込めて作成"
            }
        })

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
    
    def make_request(self, messages: List[Dict], model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 1200) -> Optional[str]:
        """Wykonanie requestu do OpenAI z obsługą błędów"""
        try:
            self._rate_limit_delay()
            
            # Policz tokeny wejściowe
            input_text = " ".join([msg["content"] for msg in messages])
            input_tokens = count_tokens(input_text, model)
            
            # i18n spinner
            spinner_label = {
                "Polski": "🤔 Przetwarzam...",
                "English": "🤔 Processing...",
                "Deutsch": "🤔 Verarbeite...",
                "Українська": "🤔 Обробляю...",
                "Français": "🤔 Traitement...",
                "Español": "🤔 Procesando...",
                "العربية": "🤔 جارٍ المعالجة...",
                "Arabski (libański dialekt)": "🤔 عم بشتغل...",
                "中文": "🤔 正在处理...",
                "日本語": "🤔 処理中..."
            }.get(st.session_state.get("interface_lang", "Polski"), "🤔 Processing...")
            with st.spinner(spinner_label):
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
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
    
    def transcribe_audio(self, file_bytes: bytes, filename: str = "audio.wav", language_code: Optional[str] = None) -> Optional[str]:
        """Transkrypcja audio w chmurze (OpenAI)"""
        try:
            self._rate_limit_delay()
            bio = io.BytesIO(file_bytes)
            bio.name = filename
            t_spinner_label = {
                "Polski": "🎤 Rozpoznaję mowę...",
                "English": "🎤 Recognizing speech...",
                "Deutsch": "🎤 Spracherkennung...",
                "Українська": "🎤 Розпізнаю мовлення...",
                "Français": "🎤 Reconnaissance vocale...",
                "Español": "🎤 Reconociendo voz...",
                "العربية": "🎤 يتعرف على الكلام...",
                "Arabski (libański dialekt)": "🎤 عم يتعرّف عالحكي...",
                "中文": "🎤 语音识别中...",
                "日本語": "🎤 音声認識中..."
            }.get(st.session_state.get("interface_lang", "Polski"), "🎤 Recognizing speech...")
            with st.spinner(t_spinner_label):
                try:
                    if language_code:
                        resp = self.client.audio.transcriptions.create(
                            model="gpt-4o-mini-transcribe",
                            file=bio,
                            language=language_code
                        )
                    else:
                        resp = self.client.audio.transcriptions.create(
                            model="gpt-4o-mini-transcribe",
                            file=bio
                        )
                except Exception:
                    # Fallback bez podpowiedzi języka
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
    
    def explain_text(self, text: str, lang: str) -> Optional[str]:
        """Wyjaśnienie tekstu w języku interfejsu"""
        # Sprawdź cache (uwzględnij język)
        cache_key = generate_cache_key(text, "explain", lang=lang)
        cached_result = get_cached_response(cache_key)
        if cached_result:
            st.info("📋 Wynik z cache")
            return cached_result
        
        # Walidacja
        is_valid, error_msg = Utils.validate_text(text)
        if not is_valid:
            st.warning(error_msg)
            return None
        
        # Ustal język odpowiedzi na podstawie języka interfejsu
        interface_to_lang = {
            "Polski": "Polish",
            "English": "English",
            "Deutsch": "German",
            "Українсьka": "Ukrainian",
            "Français": "French",
            "Español": "Spanish",
            "العربية": "Arabic",
            "Arabski (libański dialekt)": "Arabic (Lebanese dialect)",
            "中文": "Chinese",
            "日本語": "Japanese",
        }
        response_language = interface_to_lang.get(lang, "Polish")

        # Przygotuj prompt – wyraźnie wymuś język odpowiedzi
        prompt = (
            "Explain the difficult words and grammar structures in the text below. "
            "Provide short vocabulary definitions and describe the used grammar in a simple way. "
            f"IMPORTANT: Respond ONLY in {response_language}. Do not include any other language.\n\n"
            f"Text: {text}"
        )
        
        # Wykonaj request
        messages = [
            {"role": "system", "content": f"You are a language teacher. Always respond in {response_language}."},
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
        # Dostęp do etykiet dla i18n rysowanych elementów (title/labels)
        self.labels = Labels.get_labels()
    
    @st.cache_data(ttl=1800)  # Cache na 30 minut
    def generate_flashcards(self, text: str, definition_language: str) -> Optional[Dict]:
        """Generowanie fiszek z tekstu z definicjami w wybranym języku i zwracanie struktury danych - zoptymalizowane dla Cloud"""
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
        
        # Ogranicz długość tekstu dla Cloud (zapobiega timeoutom)
        max_text_length = 2000  # Zmniejszone z nieograniczonej długości
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
            st.info(f"📝 Tekst został skrócony do {max_text_length} znaków dla optymalizacji")
        
        # Przygotuj prompt (zoptymalizowany dla Cloud)
        prompt = (
            "Extract 4-6 key vocabulary items from the text. "
            f"Format: word (in original language), definition (in {definition_language}), example (in original language). "
            "Keep definitions short. Respond ONLY in JSON format:\n"
            '{"flashcards": [{"word": "term", "definition": "def", "example": "example"}]}'
        )
        
        # Wykonaj request z timeoutem
        with st.spinner("🔄 Generuję fiszki..."):
            messages = [
                {"role": "system", "content": "You are a language teacher. Respond ONLY in JSON format, no extra text."},
                {"role": "user", "content": f"Text: {text}\n\n{prompt}"}
            ]
            
            result = self.openai_handler.make_request(messages, max_tokens=800)  # Ograniczone tokeny
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
    
    @st.cache_data(ttl=3600)  # Cache na 1 godzinę
    def generate_images(self, flashcards_data: Dict, size_choice: str = "Duże (800×600)", format_choice: str = "PNG (najlepsza jakość)", quality_choice: str = "Wysoka") -> Optional[bytes]:
        """Generuje obrazy PNG z fiszkami w wybranym rozmiarze - zoptymalizowane dla Streamlit Cloud"""
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
            
            # Progress bar dla Cloud
            progress_bar = st.progress(0)
            st.info("🎨 Generuję obraz fiszek...")
            
            # Ustawienia obrazu - wybór rozmiaru (zoptymalizowane dla Cloud)
            if "Duże" in size_choice:
                card_width, card_height = 600, 450  # Zmniejszone z 800x600
                margin = 40
                font_large_size, font_medium_size, font_small_size = 24, 18, 14
            elif "Średnie" in size_choice:
                card_width, card_height = 500, 375  # Zmniejszone z 600x450
                margin = 35
                font_large_size, font_medium_size, font_small_size = 20, 16, 12
            else:  # Małe
                card_width, card_height = 350, 260  # Zmniejszone z 400x300
                margin = 25
                font_large_size, font_medium_size, font_small_size = 16, 12, 9
            
            # Ogranicz liczbę fiszek dla Cloud (mniej obciążenia)
            max_cards = 4  # Zamiast nieograniczonej liczby
            cards_per_row = 2
            cards_per_col = min(2, (len(flashcards) + 1) // 2)
            
            # Rozmiar całego obrazu
            total_width = cards_per_row * card_width + (cards_per_row + 1) * margin
            total_height = cards_per_col * card_height + (cards_per_col + 1) * margin + 80  # Zmniejszone z +100
            
            # Tworzenie obrazu
            img = Image.new('RGB', (total_width, total_height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Próba załadowania czcionki z obsługą polskich znaków (zoptymalizowane dla Cloud)
            def _load_font_with_fallback(size: int):
                candidate_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu (Streamlit Cloud)
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                    "DejaVuSans.ttf",  # bieżący katalog
                    "arial.ttf",       # Windows
                ]
                for path in candidate_paths:
                    try:
                        return ImageFont.truetype(path, size)
                    except Exception:
                        continue
                return ImageFont.load_default()

            font_large = _load_font_with_fallback(font_large_size)
            font_medium = _load_font_with_fallback(font_medium_size)
            font_small = _load_font_with_fallback(font_small_size)
            
            # Tytuł
            title = self.labels.get("Flashcards image title", {}).get(st.session_state.interface_lang, "📚 Flashcards for learning")
            title_bbox = draw.textbbox((0, 0), title, font=font_large)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (total_width - title_width) // 2
            draw.text((title_x, 15), title, fill='#1f77b4', font=font_large)
            
            # Generowanie fiszek (ograniczone do max_cards)
            for i, card in enumerate(flashcards[:max_cards]):
                if i >= cards_per_row * cards_per_col:
                    break
                
                # Pozycja fiszki
                row = i // cards_per_row
                col = i % cards_per_row
                
                x = margin + col * (card_width + margin)
                y = 70 + row * (card_height + margin)  # Zmniejszone z 100
                
                # Rysowanie ramki fiszki
                draw.rectangle([x, y, x + card_width, y + card_height], 
                             outline='#1f77b4', width=2, fill='#f8f9fa')  # Zmniejszone z width=3
                
                # Linia podziału
                draw.line([x, y + card_height//2, x + card_width, y + card_height//2], 
                         fill='#ff7f0e', width=1)  # Zmniejszone z width=2
                
                # Słówko / Definicja / Przykład – i18n (wyrównanie lewym marginesem)
                left_margin = x + 15  # Zmniejszone z 20
                word_label = self.labels.get("Flashcard label - word", {}).get(st.session_state.interface_lang, "WORD:")
                word = card.get("word", "")[:25]  # Ograniczone z 30
                draw.text((left_margin, y + 15), f"{word_label} {word}", fill='#333', font=font_medium)  # Zmniejszone z y + 20
                
                # Definicja (niżej, pod linią, w jednej linii)
                def_label = self.labels.get("Flashcard label - definition", {}).get(st.session_state.interface_lang, "DEFINITION:")
                definition = card.get("definition", "")[:50]  # Ograniczone z 60
                # Linia podziału jest na y + card_height//2, więc ustaw tekst znacznie poniżej i wyrównaj do lewego marginesu
                def_y = y + card_height//2 + 15  # Zmniejszone z +20
                draw.text((left_margin, def_y), f"{def_label} {definition}", fill='#333', font=font_small)
                
                # Przykład (niżej, jedna linia)
                ex_label = self.labels.get("Flashcard label - example", {}).get(st.session_state.interface_lang, "EXAMPLE:")
                example = card.get("example", "")[:60]  # Ograniczone z 80
                ex_y = def_y + 20  # Zmniejszone z +28
                draw.text((left_margin, ex_y), f"{ex_label} {example}", fill='#666', font=font_small)
            
            # Konwersja do bytes z optymalizacją
            bio = io.BytesIO()
            if "JPG" in format_choice:
                img.save(bio, format='JPEG', quality=85, optimize=True)  # Zmniejszone z 100
            else:
                img.save(bio, format='PNG', optimize=True)
            bio.seek(0)
            
            # Zakończenie progress bar
            progress_bar.progress(100)
            st.success("✅ Obraz wygenerowany pomyślnie!")
            
            return bio.getvalue()
            
        except Exception as e:
            st.error(f"❌ Błąd generowania obrazów: {str(e)}")
            return None

    # --- Ćwiczenia wymowy (generowanie słów i analiza) ---
    def generate_practice_words(self, language: str, practice_type: str):
        try:
            prompts = {
                "Słowa podstawowe": f"Generate 5 basic words in {language} with phonetic transcription. Format: Word - Transcription - Meaning in Polish",
                "Zwroty codzienne": f"Generate 5 common daily phrases in {language} with phonetic transcription. Format: Phrase - Transcription - Meaning in Polish",
                "Liczby": f"Generate numbers 1-10 in {language} with phonetic transcription. Format: Number - Transcription - Meaning in Polish",
                "Kolory": f"Generate 8 basic colors in {language} with phonetic transcription. Format: Color - Transcription - Meaning in Polish",
                "Członkowie rodziny": f"Generate 8 family members in {language} with phonetic transcription. Format: Family member - Transcription - Meaning in Polish",
            }
            prompt = prompts.get(practice_type, prompts["Słowa podstawowe"])
            messages = [
                {"role": "system", "content": f"Jesteś nauczycielem języka {language}. Generujesz słowa do ćwiczenia wymowy."},
                {"role": "user", "content": prompt},
            ]
            result = self.openai_handler.make_request(messages)
            if result:
                st.sidebar.success(self.labels["Success: words generated"][st.session_state.interface_lang])
                # Zlokalizowana etykieta typu ćwiczenia
                display_type_key = f"Opt - {practice_type}"
                display_type = self.labels.get(display_type_key, {}).get(st.session_state.interface_lang, practice_type)
                # Zapamiętaj wynik, aby był widoczny po każdej rerunie
                st.session_state.practice_words_result = result
                st.session_state.practice_words_display_type = display_type
                st.session_state.practice_words_language = language
                st.session_state.scroll_to_practice = True
                st.sidebar.markdown(f"""
                <div style=\"background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #6f42c1;\">
                    <h4 style=\"margin: 0 0 15px 0; color: #6f42c1;\">📚 {display_type} ({language}):</h4>
                    <div style=\"font-size: 16px; line-height: 1.6; margin: 0;\">{result}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.error(self.labels["Error: words not generated"][st.session_state.interface_lang])
        except Exception as e:
            st.sidebar.error(f"{self.labels['Error: words generation exception'][st.session_state.interface_lang]} {e}")

    def analyze_pronunciation(self, language: str, recorded_text: str):
        try:
            prompt = f"""
            Przeanalizuj wymowę użytkownika w języku {language}.
            Nagrany tekst: "{recorded_text}"
            Oceń:
            1. Poprawność wymowy (1-10)
            2. Główne błędy
            3. Wskazówki do poprawy
            4. Ćwiczenia do praktyki
            Odpowiedz w formacie:
            **Ocena:** X/10
            **Błędy:** [lista]
            **Wskazówki:** [lista]
            **Ćwiczenia:** [lista]
            """
            messages = [
                {"role": "system", "content": f"Jesteś ekspertem od wymowy języka {language}."},
                {"role": "user", "content": prompt},
            ]
            result = self.openai_handler.make_request(messages)
            if result:
                st.success(self.labels["Success: pronunciation analyzed"][st.session_state.interface_lang])
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #17a2b8;">
                    <h4 style="margin: 0 0 15px 0; color: #17a2b8;">{self.labels['Pronunciation analysis'][st.session_state.interface_lang]}</h4>
                    <div style="font-size: 16px; line-height: 1.6; margin: 0;">{result}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(self.labels["Error: pronunciation not analyzed"][st.session_state.interface_lang])
        except Exception as e:
            st.error(f"{self.labels['Error: pronunciation exception'][st.session_state.interface_lang]} {e}")

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
    
    def render_sidebar(self, lang: str):
        """Renderowanie sidebar"""
        st.sidebar.title(self.labels["Ustawienia"][lang])
        
        # Wybór motywu
        st.sidebar.subheader(self.labels["Motyw"][lang])
        bg_color = st.sidebar.radio(
            self.labels["Kolor tła"][lang],
            [self.labels["Jasny"][lang], self.labels["Ciemny"][lang]],
            index=0
        )
        
        # Informacje o aplikacji
        st.sidebar.markdown("---")
        st.sidebar.subheader(self.labels["O aplikacji"][lang])
        st.sidebar.markdown(self.labels["About content"][lang])
        
        # Sekcja ćwiczeń przeniesiona na ekran główny
        st.sidebar.markdown("---")
        
        # Statystyki
        if 'request_count' not in st.session_state:
            st.session_state.request_count = 0
        
        st.sidebar.markdown(f"{self.labels['Liczba requestów'][lang]}: {st.session_state.request_count}")
        
        return lang, bg_color

    def generate_practice_words(self, language: str, practice_type: str):
        """Generowanie słów do ćwiczenia wymowy (Cloud)"""
        try:
            prompts = {
                "Słowa podstawowe": (
                    f"Generate 7 basic yet varied words in {language} with phonetic transcription. "
                    f"Avoid repeating items from the provided 'previous_items' list. "
                    f"Prefer diversity across parts of speech. "
                    f"Randomize selection on each call. "
                    f"Format each item strictly as: Word - Transcription - Meaning in Polish"
                ),
                "Zwroty codzienne": (
                    f"Generate 7 common daily phrases in {language} with phonetic transcription. "
                    f"Avoid repeating items from 'previous_items'. Randomize selection. "
                    f"Format: Phrase - Transcription - Meaning in Polish"
                ),
                "Liczby": (
                    f"Generate 10 numbers 1-10 in {language} with phonetic transcription. "
                    f"If 'previous_items' overlaps, shuffle order and provide alternative usage examples in parentheses. "
                    f"Format: Number - Transcription - Meaning in Polish"
                ),
                "Kolory": (
                    f"Generate 10 basic colors in {language} with phonetic transcription. "
                    f"Avoid repeating items from 'previous_items'. Randomize selection. "
                    f"Format: Color - Transcription - Meaning in Polish"
                ),
                "Członkowie rodziny": (
                    f"Generate 10 family members in {language} with phonetic transcription. "
                    f"Avoid repeating items from 'previous_items'. Randomize selection. "
                    f"Format: Family member - Transcription - Meaning in Polish"
                ),
            }
            prompt = prompts.get(practice_type, prompts["Słowa podstawowe"])
            history_key = f"practice_history::{language}::{practice_type}"
            prev_items = st.session_state.get(history_key, [])
            messages = [
                {"role": "system", "content": f"You are a language teacher for {language}. Provide varied, non-repeating items with phonetic transcription. Keep output concise."},
                {"role": "user", "content": prompt + (f"\nprevious_items: {prev_items}" if prev_items else "")},
            ]
            result = self.openai_handler.make_request(messages, temperature=0.9)
            if result:
                st.success(self.labels["Success: words generated"][st.session_state.interface_lang])
                display_type_key = f"Opt - {practice_type}"
                display_type = self.labels.get(display_type_key, {}).get(st.session_state.interface_lang, practice_type)
                # Zapamiętaj wynik i ustaw flagę przewinięcia
                st.session_state.practice_words_result = result
                st.session_state.practice_words_display_type = display_type
                st.session_state.practice_words_language = language
                st.session_state.scroll_to_practice = True
                # Aktualizacja historii wygenerowanych elementów
                try:
                    new_items = []
                    for line in result.splitlines():
                        if " - " in line:
                            head = line.split(" - ", 1)[0].strip()
                            if head and head not in prev_items:
                                new_items.append(head)
                    st.session_state[history_key] = (prev_items + new_items)[-200:]
                except Exception:
                    pass
                st.rerun()
            else:
                st.error(self.labels["Error: words not generated"][st.session_state.interface_lang])
        except Exception as e:
            st.error(f"{self.labels['Error: words generation exception'][st.session_state.interface_lang]} {e}")

    def analyze_pronunciation(self, language: str, recorded_text: str):
        """Analiza wymowy (Cloud)"""
        try:
            prompt = f"""
            Przeanalizuj wymowę użytkownika w języku {language}.
            Nagrany tekst: "{recorded_text}"
            Oceń:
            1. Poprawność wymowy (1-10)
            2. Główne błędy
            3. Wskazówki do poprawy
            4. Ćwiczenia do praktyki
            Odpowiedz w formacie:
            **Ocena:** X/10
            **Błędy:** [lista]
            **Wskazówki:** [lista]
            **Ćwiczenia:** [lista]
            """
            messages = [
                {"role": "system", "content": f"Jesteś ekspertem od wymowy języka {language}."},
                {"role": "user", "content": prompt},
            ]
            result = self.openai_handler.make_request(messages)
            if result:
                st.success(self.labels["Success: pronunciation analyzed"][st.session_state.interface_lang])
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #17a2b8;">
                    <h4 style="margin: 0 0 15px 0; color: #17a2b8;">{self.labels['Pronunciation analysis'][st.session_state.interface_lang]}</h4>
                    <div style="font-size: 16px; line-height: 1.6; margin: 0;">{result}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(self.labels["Error: pronunciation not analyzed"][st.session_state.interface_lang])
        except Exception as e:
            st.error(f"{self.labels['Error: pronunciation exception'][st.session_state.interface_lang]} {e}")
    
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
        
        text_key = f"translation_text_v{st.session_state.translation_text_version}"
        text = st.text_area(
            self.labels["Wprowadź tekst tutaj:"][lang],
            value=initial_text,
            height=150,
            placeholder=self.labels["Placeholder: tłumaczenie"][lang],
            key=text_key
        )
        # Wyczyść tekst – przycisk pod polem (bez bezpośredniej modyfikacji klucza istniejącego widgetu)
        clear_col, _ = st.columns([1, 3])
        with clear_col:
            if st.button(self.labels["Wyczyść tekst"][lang], key="translation_clear_btn", use_container_width=True):
                st.session_state.recorded_translation_text = ""
                st.session_state.translation_text_version += 1
                st.rerun()
        
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
                interface_lang_hints = {
                    "Polski": "pl-PL",
                    "English": "en-US",
                    "Deutsch": "de-DE",
                    "Українська": "uk-UA",
                    "Français": "fr-FR",
                    "Español": "es-ES",
                    "العربية": "ar",
                    "Arabski (libański dialekt)": "ar",
                    "中文": "zh-CN",
                    "日本語": "ja-JP"
                }
                lang_hint = interface_lang_hints.get(lang, None)
                text_from_mic = self.openai_handler.transcribe_audio(audio_bytes, "mic.wav", language_code=lang_hint)
                if text_from_mic:
                    st.session_state.recorded_translation_text = text_from_mic
                    # Zresetuj widget przez zmianę klucza (inkrementacja wersji)
                    st.session_state.mic_widget_version += 1
                    st.success(self.labels["Success: mic recognized"][lang])
                    st.rerun()
                else:
                    st.warning(self.labels["Warn: mic not recognized"][lang])
        with col2:
            file_key = f"translation_audio_upload_v{st.session_state.file_widget_version}"
            audio_file = st.file_uploader(
                self.labels["Wczytaj plik audio"][lang],
                type=["wav", "mp3", "m4a"],
                key=file_key
            )
            if audio_file is not None:
                uploaded_bytes = audio_file.getvalue()
                interface_lang_hints = {
                    "Polski": "pl-PL",
                    "English": "en-US",
                    "Deutsch": "de-DE",
                    "Українська": "uk-UA",
                    "Français": "fr-FR",
                    "Español": "es-ES",
                    "العربية": "ar",
                    "Arabski (libański dialekt)": "ar",
                    "中文": "zh-CN",
                    "日本語": "ja-JP"
                }
                lang_hint = interface_lang_hints.get(lang, None)
                text_from_file = self.openai_handler.transcribe_audio(uploaded_bytes, audio_file.name, language_code=lang_hint)
                if text_from_file:
                    st.session_state.recorded_translation_text = text_from_file
                    # Zresetuj widget przez zmianę klucza (inkrementacja wersji)
                    st.session_state.file_widget_version += 1
                    st.success(self.labels["Success: file recognized"][lang])
                    st.rerun()
                else:
                    st.warning(self.labels["Warn: file not recognized"][lang])
        
        st.markdown("---")
        
        # Opcje tłumaczenia
        col1, col2 = st.columns([1, 1])
        with col1:
            correct_errors = st.checkbox(self.labels["Popraw błędy przed tłumaczeniem"][lang], value=False, help=self.labels["Help: Popraw błędy przed tłumaczeniem"][lang])
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
                        <h3 style="margin: 0 0 20px 0; color: #1f77b4; font-size: 24px; font-weight: 700; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">✨ {self.labels['Result'][lang]} ({target_lang}):</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result["translation"]:
                        # Sprawdź czy to jest wynik z poprawą błędów
                        if correct_errors and ("Wykryty język:" in result["translation"] or "Poprawiony tekst:" in result["translation"]):
                            # Wyświetl w czterech kolumnach jedna pod drugą
                            
                            # Kolumna 1: Wykryty język (i18n)
                            st.markdown(f"""
                            <div style="background-color: #e3f2fd; padding: 20px; border-radius: 15px; border-left: 8px solid #2196f3; margin: 0 0 20px 0; width: 100%; box-sizing: border-box; min-height: 120px;">
                                <h4 style="margin: 0 0 15px 0; color: #2196f3; font-size: 18px; font-weight: 600; text-align: left;">{self.labels['Detected language'][lang]}</h4>
                                <div style="font-size: 16px; line-height: 1.6; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">
                                    {self._extract_section(result["translation"], "Wykryty język:")}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Kolumna 2: Poprawiony tekst (i18n)
                            st.markdown(f"""
                            <div style="background-color: #f3e5f5; padding: 20px; border-radius: 15px; border-left: 8px solid #9c27b0; margin: 0 0 20px 0; width: 100%; box-sizing: border-box; min-height: 120px;">
                                <h4 style="margin: 0 0 15px 0; color: #9c27b0; font-size: 18px; font-weight: 600; text-align: left;">{self.labels['Corrected text'][lang]}</h4>
                                <div style="font-size: 16px; line-height: 1.6; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">
                                    {self._extract_section(result["translation"], "Poprawiony tekst:")}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Kolumna 3: Tłumaczenie (i18n)
                            st.markdown(f"""
                            <div style="background-color: #e8f5e8; padding: 20px; border-radius: 15px; border-left: 8px solid #4caf50; margin: 0 0 20px 0; width: 100%; box-sizing: border-box; min-height: 120px;">
                                <h4 style="margin: 0 0 15px 0; color: #4caf50; font-size: 18px; font-weight: 600; text-align: left;">📝 {self.labels['Translation'][lang]}</h4>
                                <div style="font-size: 16px; line-height: 1.6; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">
                                    {self._extract_section(result["translation"], "Tłumaczenie na")}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Kolumna 4: Transkrypcja (i18n)
                            st.markdown(f"""
                            <div style="background-color: #fff3e0; padding: 20px; border-radius: 15px; border-left: 8px solid #ff9800; margin: 0 0 20px 0; width: 100%; box-sizing: border-box; min-height: 120px;">
                                <h4 style="margin: 0 0 15px 0; color: #ff9800; font-size: 18px; font-weight: 600; text-align: left;">{self.labels['Transcription'][lang]}</h4>
                                <div style="font-size: 16px; line-height: 1.6; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">
                                    {self._extract_section(result["translation"], "Transkrypcja:")}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Standardowe wyświetlanie tłumaczenia w jednej kolumnie
                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #1f77b4; margin: 0; width: 100%; box-sizing: border-box;">
                                <h4 style="margin: 0 0 20px 0; color: #1f77b4; font-size: 20px; font-weight: 600; text-align: left;">📝 {self.labels['Translation'][lang]}</h4>
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
                                <h4 style="margin: 0 0 15px 0; color: #495057; font-size: 18px; font-weight: 600; text-align: left;">{self.labels['Listen translation'][lang]}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            st.audio(audio_content, format="audio/mp3")
                            

            else:
                st.warning(self.labels["Warn: enter text to translate"][lang])
    
    def render_explanation_section(self, lang: str):
        """Renderowanie sekcji wyjaśnień"""
        st.header(self.labels["Wyjaśnienia słów i gramatyki"][lang])
        
        explain_text = st.text_area(
            self.labels["Wpisz zdanie lub tekst do wyjaśnienia:"][lang],
            height=120,
            placeholder=self.labels["Placeholder: wyjaśnienia"][lang],
            key="explanation_text"
        )
        # Wyczyść tekst – przycisk pod polem
        clear_col, _ = st.columns([1, 3])
        with clear_col:
            st.button(
                self.labels["Wyczyść tekst"][lang],
                key="explanation_clear_btn",
                use_container_width=True,
                on_click=lambda: st.session_state.__setitem__("explanation_text", ""),
            )
        
        if st.button(
            self.labels["Wyjaśnij słowa i gramatykę"][lang],
            type="secondary",
            use_container_width=True
        ):
            if explain_text:
                st.session_state.request_count += 1
                explanation = self.explanation_manager.explain_text(explain_text, lang)
                
                if explanation:
                    st.markdown("---")
                    # Wyświetl wyjaśnienia w lepszym formacie
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #28a745; margin: 0; width: 100%; box-sizing: border-box;">
                        <h4 style="margin: 0 0 20px 0; color: #28a745; font-size: 20px; font-weight: 600; text-align: left;">📚 {self.labels['Wyjaśnienia słów i gramatyki'][lang]}:</h4>
                        <div style="font-size: 18px; line-height: 1.8; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">{explanation}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(self.labels["Warn: enter text to explain"][lang])
    
    def render_style_section(self, lang: str):
        # Jeśli włączona jest opcja poprawy błędów przed tłumaczeniem, pokazujemy sekcję stylistyki tylko gdy użytkownik faktycznie jej potrzebuje
        # (nie ukrywamy twardo, ale zostawiamy jasny podtytuł)
        """Renderowanie sekcji stylistyki"""
        st.header(self.labels["Ładna wersja wypowiedzi – poprawa stylistyki"][lang])
        st.caption(self.labels["Style caption"][lang])
        
        style_text = st.text_area(
            self.labels["Wpisz tekst do poprawy stylistycznej:"][lang],
            height=120,
            placeholder=self.labels["Placeholder: stylistyka"][lang],
            key="style_text"
        )
        # Wyczyść tekst – przycisk pod polem
        clear_style_col, _ = st.columns([1, 3])
        with clear_style_col:
            st.button(
                self.labels["Wyczyść tekst"][lang],
                key="style_clear_btn",
                use_container_width=True,
                on_click=lambda: st.session_state.__setitem__("style_text", ""),
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
                        <h4 style="margin: 0 0 20px 0; color: #ffc107; font-size: 20px; font-weight: 600; text-align: left;">✨ {self.labels['Ładna wersja wypowiedzi – poprawa stylistyki'][lang]}:</h4>
                        <div style="font-size: 18px; line-height: 1.8; margin: 0; font-weight: 500; text-align: left; word-wrap: break-word; white-space: normal; overflow-wrap: break-word;">{nice_version}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(self.labels["Warn: enter text to improve"][lang])
    

    
    def render_flashcard_section(self, lang: str):
        """Renderowanie sekcji fiszek"""
        st.header(self.labels["Fiszki ze słówek do nauki"][lang])
        
        flashcard_text = st.text_area(
            self.labels["Wpisz tekst, z którego chcesz wygenerować fiszki:"][lang],
            height=120,
            placeholder=self.labels["Placeholder: fiszki"][lang],
            key="flashcard_text"
        )
        # Wyczyść tekst – przycisk pod polem
        clear_flash_col, _ = st.columns([1, 3])
        with clear_flash_col:
            st.button(
                self.labels["Wyczyść tekst"][lang],
                key="flashcard_clear_btn",
                use_container_width=True,
                on_click=lambda: st.session_state.__setitem__("flashcard_text", ""),
            )
        
        # Wybór języka definicji (interfejs / lista)
        st.caption(self.labels["Wybierz język definicji fiszek"][lang])
        definition_lang_choice = st.selectbox(
            self.labels["Język definicji"][lang],
            [
                self.labels["Język interfejsu (opcja)"][lang],
                "Polish",
                "English",
                "German",
                "French",
                "Spanish",
                "Italian",
                "Arabic",
                "Chinese",
                "Japanese",
            ],
            index=0,
            key="flashcards_definition_lang"
        )
        # Mapowanie języka interfejsu -> język definicji (angielskie nazwy dla spójności promptów)
        interface_to_lang = {
            "Polski": "Polish",
            "English": "English",
            "Deutsch": "German",
            "Українсьka": "Ukrainian",
            "Français": "French",
            "Español": "Spanish",
            "العربية": "Arabic",
            "Arabski (libański dialekt)": "Arabic (Lebanese dialect)",
            "中文": "Chinese",
            "日本語": "Japanese",
        }
        effective_definition_lang = interface_to_lang.get(lang, "Polish") if definition_lang_choice == self.labels["Język interfejsu (opcja)"][lang] else definition_lang_choice

        if st.button(
            self.labels["Wygeneruj fiszki"][lang],
            type="secondary",
            use_container_width=True
        ):
            if flashcard_text:
                st.session_state.request_count += 1
                flashcards_data = self.flashcard_manager.generate_flashcards(flashcard_text, effective_definition_lang)
                
                if flashcards_data and "flashcards" in flashcards_data:
                    # Zachowaj dane fiszek w stanie i przejdź do stałej sekcji podglądu
                    st.session_state.flashcards_data = flashcards_data
                    st.session_state.flashcards_image = None
                    st.rerun()
                    st.markdown("---")
                    # Wyświetl fiszki w lepszym formacie
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #6f42c1; margin: 0; width: 100%; box-sizing: border-box;">
                    <h4 style="margin: 0 0 20px 0; color: #6f42c1; font-size: 20px; font-weight: 600; text-align: left;">📖 {self.labels['Generated flashcards'][lang] if 'Generated flashcards' in self.labels else self.labels['Fiszki ze słówek do nauki'][lang]}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Sprawdź czy to nie są fiszki z błędami
                    if len(flashcards_data["flashcards"]) == 1 and flashcards_data["flashcards"][0].get("word", "").startswith("Błąd"):
                        st.error("❌ Wystąpił błąd podczas generowania fiszek. Spróbuj ponownie.")
                        st.info("💡 **Wskazówka:** Upewnij się, że tekst jest w języku, który chcesz przetłumaczyć.")
                        return
                    
                    # Wyświetl fiszki w ładnym formacie (i18n)
                    for i, card in enumerate(flashcards_data["flashcards"], 1):
                        expander_title = self.labels.get("Flashcard expander title", {}).get(lang, "Flashcard")
                        word_label = self.labels.get("Flashcard label - word", {}).get(lang, "WORD:")
                        def_label = self.labels.get("Flashcard label - definition", {}).get(lang, "DEFINITION:")
                        ex_label = self.labels.get("Flashcard label - example", {}).get(lang, "EXAMPLE:")
                        missing_word = self.labels.get("Missing - word", {}).get(lang, "N/A")
                        with st.expander(f"🃏 {expander_title} {i}: {card.get('word', missing_word)}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**🔤 {word_label}** {card.get('word', missing_word)}")
                                st.markdown(f"**📝 {def_label}** {card.get('definition', missing_word)}")
                            with col2:
                                st.markdown(f"**💡 {ex_label}** {card.get('example', missing_word)}")
                    
                    # Generuj obrazy fiszek
                    st.markdown("---")
                    # Wyświetl nagłówek w lepszym formacie
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #6f42c1; margin: 0; width: 100%; box-sizing: border-box;">
                        <h4 style="margin: 0 0 20px 0; color: #6f42c1; font-size: 20px; font-weight: 600; text-align: left;">🖼️ {self.labels['Download flashcards to print'][lang]}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Wybór formatu (z kluczami, bez natychmiastowego generowania)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        format_choice = st.selectbox(
                            self.labels["Select format"][lang],
                            [self.labels["Format - PNG best"][lang], self.labels["Format - JPG smaller"][lang], self.labels["Format - PDF print"][lang]],
                            index=0,
                            key="flashcards_format"
                        )
                    with col2:
                        quality_choice = st.selectbox(
                            self.labels["Quality"][lang],
                            [self.labels["Quality - High"][lang], self.labels["Quality - Medium"][lang], self.labels["Quality - Low"][lang]],
                            index=0,
                            key="flashcards_quality"
                        )
                    with col3:
                        size_choice = st.selectbox(
                            self.labels["Flashcard size"][lang],
                            [self.labels["Size - Large"][lang], self.labels["Size - Medium"][lang], self.labels["Size - Small"][lang]],
                            index=0,
                            key="flashcards_size"
                        )

                    # Przycisk generowania obrazu (unikamy ciężkiego przeliczenia przy każdej zmianie selecta)
                    if st.button(self.labels.get("Generate image", {}).get(lang, "🖼️ Generate image"), key="flashcards_generate_image_btn"):
                        image_data = self.flashcard_manager.generate_images(flashcards_data, size_choice, format_choice, quality_choice)
                        st.session_state.flashcards_image = {
                            "data": image_data,
                            "format_choice": format_choice,
                            "quality_choice": quality_choice,
                            "size_choice": size_choice,
                        }

                    image_state = st.session_state.get("flashcards_image")
                    image_data = image_state.get("data") if image_state else None

                    if image_data:
                        st.success(self.labels.get("Image generated ok", {}).get(lang, "✅ Image generated successfully!"))
                        
                        # Podgląd obrazu
                        st.markdown(f"""
                        <div style=\"background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #6f42c1; margin: 0; width: 100%; box-sizing: border-box;\">
                            <h4 style=\"margin: 0 0 20px 0; color: #6f42c1; font-size: 20px; font-weight: 600; text-align: left;\">{self.labels.get('Flashcards preview', {}).get(lang, '👀 Flashcards preview:')}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(image_data, caption=self.labels.get("Flashcards preview", {}).get(lang, "👀 Flashcards preview:"), use_container_width=True)
                        
                        # Przyciski pobierania
                        col1, col2 = st.columns(2)
                        with col1:
                            # Określenie formatu i rozszerzenia pliku
                            current_format_choice = image_state.get("format_choice") if image_state else format_choice
                            if current_format_choice and "JPG" in current_format_choice:
                                file_extension = "jpg"
                                mime_type = "image/jpeg"
                            else:
                                file_extension = "png"
                                mime_type = "image/png"
                            
                            st.download_button(
                                label=self.labels["Download flashcards"][lang],
                                data=image_data,
                                file_name=f"fiszki_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                                mime=mime_type,
                                use_container_width=True,
                                type="primary"
                            )
                        

                        
                        # Szczegółowe instrukcje (i18n)
                        expander_label = self.labels["Cutting instructions - expander"][lang]
                        with st.expander(expander_label):
                            st.markdown(self.labels["Cutting instructions - content"][lang])
                        
                        st.info(self.labels["Quick tips"][lang])
                    else:
                        st.info(self.labels.get("Flashcards preview", {}).get(lang, "👀 Flashcards preview:") + " — " + (self.labels.get("Generate image", {}).get(lang, "click 'Generate image' to preview")))
                else:
                    st.warning("Nie udało się wygenerować fiszek.")
            else:
                st.warning(self.labels["Warn: enter text to generate flashcards"][lang])

        # Stała sekcja podglądu i generowania obrazów (utrzymywana między rerunami)
        if st.session_state.get("flashcards_data"):
            flashcards_data = st.session_state.flashcards_data
            st.markdown("---")
            st.markdown(f"""
            <div style=\"background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #6f42c1; margin: 0; width: 100%; box-sizing: border-box;\">\n                <h4 style=\"margin: 0 0 20px 0; color: #6f42c1; font-size: 20px; font-weight: 600; text-align: left;\">📖 {self.labels['Generated flashcards'][lang] if 'Generated flashcards' in self.labels else self.labels['Fiszki ze słówek do nauki'][lang]}</h4>
            </div>
            """, unsafe_allow_html=True)

            # Lista fiszek (i18n)
            for i, card in enumerate(flashcards_data.get("flashcards", []), 1):
                expander_title = self.labels.get("Flashcard expander title", {}).get(lang, "Flashcard")
                word_label = self.labels.get("Flashcard label - word", {}).get(lang, "WORD:")
                def_label = self.labels.get("Flashcard label - definition", {}).get(lang, "DEFINITION:")
                ex_label = self.labels.get("Flashcard label - example", {}).get(lang, "EXAMPLE:")
                missing_word = self.labels.get("Missing - word", {}).get(lang, "N/A")
                with st.expander(f"🃏 {expander_title} {i}: {card.get('word', missing_word)}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**🔤 {word_label}** {card.get('word', missing_word)}")
                        st.markdown(f"**📝 {def_label}** {card.get('definition', missing_word)}")
                    with col2:
                        st.markdown(f"**💡 {ex_label}** {card.get('example', missing_word)}")

            st.markdown("---")
            st.markdown(f"""
            <div style=\"background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #6f42c1; margin: 0; width: 100%; box-sizing: border-box;\">\n                <h4 style=\"margin: 0 0 20px 0; color: #6f42c1; font-size: 20px; font-weight: 600; text-align: left;\">🖼️ {self.labels['Download flashcards to print'][lang]}</h4>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                format_choice = st.selectbox(
                    self.labels["Select format"][lang],
                    [self.labels["Format - PNG best"][lang], self.labels["Format - JPG smaller"][lang], self.labels["Format - PDF print"][lang]],
                    index=0,
                    key="flashcards_format"
                )
            with col2:
                quality_choice = st.selectbox(
                    self.labels["Quality"][lang],
                    [self.labels["Quality - High"][lang], self.labels["Quality - Medium"][lang], self.labels["Quality - Low"][lang]],
                    index=0,
                    key="flashcards_quality"
                )
            with col3:
                size_choice = st.selectbox(
                    self.labels["Flashcard size"][lang],
                    [self.labels["Size - Large"][lang], self.labels["Size - Medium"][lang], self.labels["Size - Small"][lang]],
                    index=0,
                    key="flashcards_size"
                )

            if st.button(self.labels.get("Generate image", {}).get(lang, "🖼️ Generate image"), key="flashcards_generate_image_btn"):
                img = self.flashcard_manager.generate_images(flashcards_data, size_choice, format_choice, quality_choice)
                st.session_state.flashcards_image = {
                    "data": img,
                    "format_choice": format_choice,
                    "quality_choice": quality_choice,
                    "size_choice": size_choice,
                }

            image_state = st.session_state.get("flashcards_image")
            image_data = image_state.get("data") if image_state else None

            if image_data:
                st.success(self.labels.get("Image generated ok", {}).get(lang, "✅ Image generated successfully!"))
                st.markdown(f"""
                <div style=\"background-color: #f0f2f6; padding: 25px; border-radius: 15px; border-left: 8px solid #6f42c1; margin: 0; width: 100%; box-sizing: border-box;\">\n                    <h4 style=\"margin: 0 0 20px 0; color: #6f42c1; font-size: 20px; font-weight: 600; text-align: left;\">{self.labels.get('Flashcards preview', {}).get(lang, '👀 Flashcards preview:')}</h4>
                </div>
                """, unsafe_allow_html=True)
                st.image(image_data, caption=self.labels.get("Flashcards preview", {}).get(lang, "👀 Flashcards preview:"), use_container_width=True)
                dl_col, _ = st.columns(2)
                with dl_col:
                    current_format_choice = image_state.get("format_choice") if image_state else format_choice
                    if current_format_choice and "JPG" in current_format_choice:
                        file_extension = "jpg"
                        mime_type = "image/jpeg"
                    else:
                        file_extension = "png"
                        mime_type = "image/png"
                    st.download_button(
                        label=self.labels["Download flashcards"][lang],
                        data=image_data,
                        file_name=f"fiszki_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                        mime=mime_type,
                        use_container_width=True,
                        type="primary",
                        key="flashcards_download"
                    )
                with st.expander(self.labels["Cutting instructions - expander"][lang]):
                    st.markdown(self.labels["Cutting instructions - content"][lang])
                st.info(self.labels["Quick tips"][lang])
            else:
                st.info(self.labels.get("Flashcards preview", {}).get(lang, "👀 Flashcards preview:") + " — " + (self.labels.get("Generate image", {}).get(lang, "click 'Generate image' to preview")))
    
    def render_footer(self, lang: str):
        """Renderowanie stopki"""
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>{self.labels['Footer tagline'][lang]}</p>
            <p>{self.labels['Footer made with'][lang]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metoda render_pronunciation_practice_section została usunięta dla kompatybilności ze Streamlit Cloud
    
    # Metody związane z ćwiczeniem wymowy zostały usunięte dla kompatybilności ze Streamlit Cloud
    
    def run(self):
        """Uruchomienie aplikacji"""
        # Ekran startowy: wybór języka interfejsu i klucz API
        if not st.session_state.setup_done:
            # Wybór języka interfejsu na głównej stronie (domyślnie PL, ale natychmiast przełącza UI)
            interface_lang = st.selectbox(
                "🌐 Język interfejsu / Interface language",
                ["Polski", "English", "Deutsch", "Українська", "Français", "Español", "العربية", "Arabski (libański dialekt)", "中文", "日本語"],
                index=["Polski", "English", "Deutsch", "Українська", "Français", "Español", "العربية", "Arabski (libański dialekt)", "中文", "日本語"].index(st.session_state.interface_lang),
                key="setup_interface_lang"
            )
            if interface_lang != st.session_state.interface_lang:
                st.session_state.interface_lang = interface_lang
                st.rerun()

            lang = st.session_state.interface_lang
            # Nagłówek po wybraniu języka
            st.markdown(f"""
            <div style=\"margin: 0; width: 100%; box-sizing: border-box;\">
                <h1 style=\"margin: 0 0 24px 0; color: #1f77b4; font-size: 32px; font-weight: 700; text-align: left;\">{self.labels["Tłumacz wielojęzyczny"][lang]}</h1>
            </div>
            """, unsafe_allow_html=True)

            # Klucz API na głównej stronie (zamiast w sidebarze)
            api_key_placeholder = "sk-..."
            api_key_label = "🔑 Wprowadź swój klucz API OpenAI:" if lang == "Polski" else "🔑 Enter your OpenAI API key:"
            proceed_label = "✅ Rozpocznij" if lang == "Polski" else "✅ Start"
            api_key_val = st.text_input(api_key_label, type="password", placeholder=api_key_placeholder)
            proceed = st.button(proceed_label)
            if proceed:
                if not api_key_val or not api_key_val.startswith("sk-"):
                    st.error("❌ Nieprawidłowy klucz API (powinien zaczynać się od 'sk-')" if lang == "Polski" else "❌ Invalid API key (must start with 'sk-')")
                    st.stop()
                st.session_state.api_key = api_key_val
                st.session_state.setup_done = True
                st.rerun()
            st.stop()

        # Od tego momentu UI jest w wybranym języku
        lang = st.session_state.interface_lang

        # Inicjalizacja klienta OpenAI
        self.client = get_openai_client(st.session_state.api_key)
        if not self.client:
            st.error("❌ Nie można zainicjalizować klienta OpenAI. Sprawdź klucz API.")
            st.stop()

        # Inicjalizacja menedżerów
        self.openai_handler = OpenAIHandler(self.client)
        # Rejestracja translatora dla auto‑tłumaczeń etykiet
        Labels.set_translator(self.openai_handler)
        self.translation_manager = TranslationManager(self.openai_handler)
        self.explanation_manager = ExplanationManager(self.openai_handler)
        self.style_manager = StyleManager(self.openai_handler)
        self.correction_manager = CorrectionManager(self.openai_handler)
        self.flashcard_manager = FlashcardManager(self.openai_handler)

        # Renderuj sidebar (bez języka i klucza — już ustawione)
        lang, bg_color = self.render_sidebar(lang)

        # Wyświetl statystyki użycia API
        display_usage_stats(lang, self.labels)

        # Aplikuj motyw
        self.apply_theme("Ciemny" if bg_color == self.labels["Ciemny"][lang] else "Jasny")

        # (przeniesiono render wyników do sekcji Ćwicz wymowę, tu już nie wyświetlamy)

        # Sekcje aplikacji
        self.render_translation_section(lang)
        st.markdown("---")

        self.render_explanation_section(lang)
        st.markdown("---")

        self.render_style_section(lang)
        st.markdown("---")

        self.render_flashcard_section(lang)

        # Nowa sekcja: Ćwicz wymowę (przeniesiona z sidebara)
        st.markdown("---")
        st.markdown(f"""
        <div style=\"margin: 0; width: 100%; box-sizing: border-box;\">
            <h2 style=\"margin: 0 0 20px 0; color: #495057; font-size: 24px; font-weight: 600; text-align: left;\">{self.labels['Ćwicz wymowę'][lang]}</h2>
        </div>
        """, unsafe_allow_html=True)

        col_practice_1, col_practice_2 = st.columns([1, 1])
        with col_practice_1:
            practice_lang = st.selectbox(
                self.labels["Język do ćwiczenia"][lang],
                ["English", "German", "French", "Spanish", "Italian", "Polish", "Arabic", "Chinese", "Japanese"],
                index=0,
                key="practice_language_select_main"
            )
        with col_practice_2:
            practice_type = st.selectbox(
                self.labels["Typ ćwiczenia"][lang],
                [
                    self.labels["Opt - Słowa podstawowe"][lang],
                    self.labels["Opt - Zwroty codzienne"][lang],
                    self.labels["Opt - Liczby"][lang],
                    self.labels["Opt - Kolory"][lang],
                    self.labels["Opt - Członkowie rodziny"][lang],
                ],
                index=0,
                key="practice_type_select_main"
            )

        if st.button(self.labels["Generuj słowa do ćwiczenia"][lang], use_container_width=True, key="generate_practice_main"):
            reverse_map = {
                self.labels["Opt - Słowa podstawowe"][lang]: "Słowa podstawowe",
                self.labels["Opt - Zwroty codzienne"][lang]: "Zwroty codzienne",
                self.labels["Opt - Liczby"][lang]: "Liczby",
                self.labels["Opt - Kolory"][lang]: "Kolory",
                self.labels["Opt - Członkowie rodziny"][lang]: "Członkowie rodziny",
            }
            selected_key = reverse_map.get(practice_type, "Słowa podstawowe")
            self.generate_practice_words(practice_lang, selected_key)

        # Przyciski: Wygeneruj inne (pomijając ostatnie) i Wyczyść historię
        ctrl_col1, ctrl_col2 = st.columns([1, 1])
        with ctrl_col1:
            if st.button("🔄 Wygeneruj inne", key="generate_practice_alt"):
                # Zachowaj historię, ale ponów wywołanie dla nowych propozycji
                reverse_map = {
                    self.labels["Opt - Słowa podstawowe"][lang]: "Słowa podstawowe",
                    self.labels["Opt - Zwroty codzienne"][lang]: "Zwroty codzienne",
                    self.labels["Opt - Liczby"][lang]: "Liczby",
                    self.labels["Opt - Kolory"][lang]: "Kolory",
                    self.labels["Opt - Członkowie rodziny"][lang]: "Członkowie rodziny",
                }
                selected_key = reverse_map.get(practice_type, "Słowa podstawowe")
                self.generate_practice_words(practice_lang, selected_key)
        with ctrl_col2:
            if st.button("🧹 Wyczyść historię", key="clear_practice_history"):
                # Wyczyść historię bieżącego języka i typu
                history_key = f"practice_history::{practice_lang}::{selected_key if 'selected_key' in locals() else 'Słowa podstawowe'}"
                st.session_state.pop(history_key, None)
                st.session_state.pop("practice_words_result", None)
                st.rerun()

        # Wyświetl wygenerowane słowa bezpośrednio pod przyciskiem
        if st.session_state.get("practice_words_result"):
            display_type = st.session_state.get("practice_words_display_type", "Practice words")
            language = st.session_state.get("practice_words_language", "")
            result_html = st.session_state.get("practice_words_result", "")
            st.markdown(f"""
            <div style=\"background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #6f42c1; margin: 16px 0;\">
                <h4 style=\"margin: 0 0 15px 0; color: #6f42c1;\">📚 {display_type} ({language}):</h4>
                <div style=\"font-size: 16px; line-height: 1.6; margin: 0;\">{result_html}</div>
            </div>
            """, unsafe_allow_html=True)

        # Nagrywanie i analiza na głównym ekranie z podpowiedzią językową
        mic_col, _ = st.columns([1, 1])
        with mic_col:
            practice_mic_key = f"practice_mic_main_v{st.session_state.practice_mic_version}"
            language_hints = {
                "Polish": "pl-PL",
                "Polski": "pl-PL",
                "English": "en-US",
                "German": "de-DE",
                "French": "fr-FR",
                "Spanish": "es-ES",
                "Italian": "it-IT",
                "Arabic": "ar-SA",
                "Chinese": "zh-CN",
                "Japanese": "ja-JP"
            }
            hint = language_hints.get(practice_lang, None)
            practice_mic = st.audio_input(self.labels["Nagraj wymowę"][lang], key=practice_mic_key)
            if practice_mic is not None:
                txt = self.openai_handler.transcribe_audio(practice_mic.getvalue(), "practice.wav", language_code=hint)
                if txt:
                    st.session_state.practice_text = txt
                    st.session_state.practice_mic_version += 1
                    st.success(self.labels["Rozpoznano wymowę"][lang])
                    st.rerun()

        if st.session_state.practice_text:
            st.caption(self.labels["Ostatnia rozpoznana wypowiedź:"][lang])
            st.info(st.session_state.practice_text)
            if st.button(self.labels["Analizuj wymowę"][lang], use_container_width=True, key="analyze_pronunciation_main"):
                self.analyze_pronunciation(practice_lang, st.session_state.practice_text)

        # Stopka
        self.render_footer(lang)

# Uruchomienie aplikacji
if __name__ == "__main__":
    # Inicjalizacja stanu sesji przed uruchomieniem aplikacji
    init_session_state()
    app = MultilingualApp()
    app.run()