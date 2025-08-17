import streamlit as st
from openai import OpenAI
import json
from datetime import datetime
from typing import Dict, List, Optional
import tiktoken
import io
import re
from audiorecorder import audiorecorder
from dotenv import load_dotenv
import os
import instructor
from pydantic import BaseModel
from pydub import AudioSegment  # Upewnij się, że pydub jest zaimportowany

# --- USTAWIENIA STRONY ---
st.set_page_config(
    page_title="Language Helper AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INICJALIZACJA SESJI ---
def init_session_state():
    # Załaduj zmienne środowiskowe
    load_dotenv()
    
    if 'api_key' not in st.session_state:
        # Sprawdź czy klucz jest w zmiennych środowiskowych
        env_api_key = os.getenv('OPENAI_API_KEY')
        st.session_state.api_key = env_api_key if env_api_key else ""
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = 0
    if 'total_cost' not in st.session_state:
        st.session_state.total_cost = 0.0
    if 'token_history' not in st.session_state:
        st.session_state.token_history = []
    if 'cost_history' not in st.session_state:
        st.session_state.cost_history = []
    if 'setup_done' not in st.session_state:
        st.session_state.setup_done = False
    if 'request_count' not in st.session_state:
        st.session_state.request_count = 0
    if 'flashcards_data' not in st.session_state:
        st.session_state.flashcards_data = None
    if 'flashcards_image' not in st.session_state:
        st.session_state.flashcards_image = None
    if 'recorded_audio_text' not in st.session_state:
        st.session_state.recorded_audio_text = ""
    if 'practice_text' not in st.session_state:
        st.session_state.practice_text = ""
    if 'practice_mic_version' not in st.session_state:
        st.session_state.practice_mic_version = 0
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None

# --- SIDEBAR & SETUP ---
def render_sidebar_and_setup():
    st.sidebar.title("🌍 Language Helper AI")
    st.sidebar.markdown("---")
    
    # Klucz API
    st.sidebar.markdown("### 🔑 Klucz API OpenAI")
    
    # Sprawdź czy klucz jest w zmiennych środowiskowych
    env_api_key = os.getenv('OPENAI_API_KEY')
    if env_api_key and not st.session_state.api_key:
        st.session_state.api_key = env_api_key
        st.sidebar.success("✅ Klucz API załadowany z zmiennych środowiskowych")
    
    api_key = st.sidebar.text_input(
        "Wprowadź swój klucz API OpenAI:",
        type="password",
        placeholder="sk-...",
        value=st.session_state.api_key,
        help="Twój klucz API OpenAI (zaczyna się od 'sk-') lub ustaw OPENAI_API_KEY w zmiennych środowiskowych"
    )
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    if not api_key or not api_key.startswith("sk-"):
        st.sidebar.warning("Podaj poprawny klucz API (zaczyna się od 'sk-') lub ustaw OPENAI_API_KEY w zmiennych środowiskowych.")
        st.stop()
    
    # Motyw
    st.sidebar.markdown("### 🎨 Motyw")
    theme = st.sidebar.radio("Wybierz motyw", ["Jasny", "Ciemny"], index=0)
    if theme == "Ciemny":
        st.markdown("""
        <style>
        body, .stApp {background-color: #0e1117 !important; color: #fafafa !important;}
        </style>
        """, unsafe_allow_html=True)
    
    # Statystyki
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Statystyki API")
    st.sidebar.metric("Tokeny", f"{st.session_state.total_tokens:,}")
    st.sidebar.metric("Koszt", f"${st.session_state.total_cost:.4f}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Aplikacja:**")
    st.sidebar.markdown("🌍 tłumaczenia")
    st.sidebar.markdown("📖 fiszki")
    st.sidebar.markdown("📚 wyjaśnienia")
    st.sidebar.markdown("📚 wskazówki gramatyczne")
    st.sidebar.markdown("🎤 wymowa")
    st.sidebar.markdown("🎙️ nagrywanie audio")

# --- KOSZTY I TOKENY ---
def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "tts-1": {"input": 0.015, "output": 0}
    }
    model_key = model if model in pricing else "gpt-4o"
    input_cost = (input_tokens / 1000) * pricing[model_key]["input"]
    output_cost = (output_tokens / 1000) * pricing[model_key]["output"]
    return input_cost + output_cost

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        return len(text) // 4

@st.cache_resource
def get_openai_client(api_key: str):
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Błąd inicjalizacji OpenAI: {e}")
        return None

def update_usage_stats(input_tokens: int, output_tokens: int, model: str):
    total_tokens = input_tokens + output_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)
    st.session_state.total_tokens += total_tokens
    st.session_state.total_cost += cost
    
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

# --- OPENAI HANDLER ---
class OpenAIHandler:
    def __init__(self, client: OpenAI):
        self.client = client
    
    def make_request(self, messages: List[Dict], model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 1200) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            output_text = response.choices[0].message.content
            input_text = " ".join([msg["content"] for msg in messages])
            input_tokens = count_tokens(input_text, model)
            output_tokens = count_tokens(output_text, model)
            update_usage_stats(input_tokens, output_tokens, model)
            return output_text
        except Exception as e:
            st.error(f"Błąd API OpenAI: {str(e)}")
            return None
    
    def transcribe_audio(self, file_bytes: bytes, filename: str = "audio.wav", language_code: Optional[str] = None) -> Optional[str]:
        """Transkrypcja audio w chmurze (OpenAI)"""
        try:
            # Mapowanie języków na kody ISO
            language_mapping = {
                "Polish": "pl",
                "English": "en", 
                "German": "de",
                "French": "fr",
                "Spanish": "es",
                "Italian": "it",
                "Arabic": "ar",
                "Chinese": "zh",
                "Japanese": "ja"
            }
            
            # Jeśli podano kod języka, użyj go
            if language_code:
                whisper_language = language_code
            else:
                # Automatyczne wykrywanie języka
                whisper_language = None
            
            # Sprawdź format pliku
            file_extension = filename.lower().split('.')[-1] if '.' in filename else 'wav'
            
            # Obsługiwane formaty przez OpenAI Whisper
            supported_formats = ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']
            
            if file_extension not in supported_formats:
                st.warning(f"⚠️ Format {file_extension} może nie być obsługiwany przez OpenAI Whisper. Zalecane: MP3, WAV, M4A")
            
            # Debug: sprawdź rozmiar pliku
            st.info(f"🔍 Rozmiar pliku: {len(file_bytes)} bajtów")
            st.info(f"🔍 Format pliku: {filename}")
            
            # Sprawdź czy plik nie jest pusty
            if len(file_bytes) == 0:
                st.error("❌ Plik audio jest pusty")
                return None
            
            # Sprawdź pierwsze bajty pliku (magic numbers)
            if len(file_bytes) >= 4:
                header = file_bytes[:4]
                if header.startswith(b'RIFF'):
                    st.info("✅ Plik ma poprawny nagłówek WAV")
                    # Sprawdź czy plik WAV ma poprawną strukturę
                    if len(file_bytes) < 44:  # Minimalny rozmiar nagłówka WAV
                        st.error("❌ Plik WAV jest za mały - uszkodzony nagłówek")
                        return None
                elif header.startswith(b'\xff\xfb') or header.startswith(b'ID3'):
                    st.info("✅ Plik ma poprawny nagłówek MP3")
                else:
                    st.warning(f"⚠️ Nieznany format pliku. Pierwsze bajty: {header.hex()}")
                    st.warning("⚠️ OpenAI może nie rozpoznać tego formatu")
            
            # Sprawdź czy plik nie jest za duży (limit OpenAI: 25MB)
            if len(file_bytes) > 25 * 1024 * 1024:
                st.error("❌ Plik jest za duży (max 25MB)")
                return None
            
            # Sprawdź czy plik nie jest za mały (min 1KB)
            if len(file_bytes) < 1024:
                st.warning("⚠️ Plik jest bardzo mały - może być uszkodzony")
            
            # Transkrypcja audio
            try:
                # Spróbuj z plikiem w pamięci
                st.info("🔄 Próbuję transkrypcji z plikiem w pamięci...")
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=io.BytesIO(file_bytes),
                    language=whisper_language,
                    response_format="text"
                )
            except Exception as api_error:
                st.error(f"❌ Błąd API OpenAI: {str(api_error)}")
                st.error(f"🔍 Typ błędu: {type(api_error).__name__}")
                
                # Spróbuj zapisać plik tymczasowo na dysku
                try:
                    st.info("🔄 Próbuję z plikiem tymczasowym...")
                    import tempfile
                    import os
                    
                    # Utwórz plik tymczasowy
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_file.write(file_bytes)
                        temp_file_path = temp_file.name
                    
                    st.info(f"🔍 Plik tymczasowy: {temp_file_path}")
                    st.info(f"🔍 Rozmiar pliku na dysku: {os.path.getsize(temp_file_path)} bajtów")
                    
                    # Spróbuj transkrypcji z pliku na dysku
                    with open(temp_file_path, 'rb') as file_obj:
                        response = self.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=file_obj,
                            response_format="text"
                        )
                    
                    # Usuń plik tymczasowy
                    os.unlink(temp_file_path)
                    st.success("✅ Transkrypcja z pliku tymczasowego udana!")
                    
                except Exception as file_error:
                    st.error(f"❌ Błąd z plikiem tymczasowym: {str(file_error)}")
                    
                    # Ostatnia próba - spróbuj z innymi opcjami
                    try:
                        st.info("🔄 Ostatnia próba - bez języka...")
                        response = self.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=io.BytesIO(file_bytes),
                            response_format="text"
                        )
                    except Exception as final_error:
                        st.error(f"❌ Wszystkie próby nieudane: {str(final_error)}")
                        return None
            
            # Aktualizacja statystyk użycia (Whisper nie zwraca tokenów)
            update_usage_stats(0, 0, "whisper-1")
            
            return response
            
        except Exception as e:
            st.error(f"❌ Błąd podczas transkrypcji audio: {str(e)}")
            return None

# --- MODELE PYDANTIC ---
class Flashcard(BaseModel):
    word: str
    definition: str
    example: str

class FlashcardSet(BaseModel):
    flashcards: List[Flashcard]

class GrammarTip(BaseModel):
    rule: str
    explanation: str
    examples: List[str]

class GrammarTips(BaseModel):
    tips: List[GrammarTip]

# --- AUDIO RECORDER MANAGER ---
class AudioRecorderManager:
    """Zarządzanie nagrywaniem audio używając audiorecorder"""

    def __init__(self):
        pass

    def record_audio(self) -> Optional[bytes]:
        """Nagrywanie audio z mikrofonu używając audiorecorder"""
        try:
            # Użyj audiorecorder do nagrania
            audio = audiorecorder(
                "🎙️ Kliknij aby rozpocząć nagrywanie", 
                "⏹️ Kliknij aby zatrzymać", 
                key=f"audio_recorder_{st.session_state.get('request_count', 0)}"
            )

            if audio is not None and len(audio) > 0:
                # Eksportuj dane audio do formatu WAV
                audio_bytes_io = io.BytesIO()
                audio.export(audio_bytes_io, format="wav")
                audio_bytes = audio_bytes_io.getvalue()

                # Wyświetl audio w Streamlit
                st.audio(audio_bytes, format="audio/wav")

                return audio_bytes
            else:
                st.warning("⚠️ Brak danych audio do przetworzenia.")
                return None

        except Exception as e:
            st.error(f"❌ Błąd podczas nagrywania: {str(e)}")
            return None
    
    def save_audio_to_session(self, audio_bytes: bytes):
        """Zapisuje nagrane audio w session state"""
        if audio_bytes:
            st.session_state.recorded_audio = audio_bytes
            return True
        return False

# --- TRANSLATION MANAGER ---
class TranslationManager:
    def __init__(self, openai_handler: OpenAIHandler):
        self.openai_handler = openai_handler
    
    def translate_text(self, text: str, target_language: str, correct_errors: bool = False, improve_style: bool = False) -> Optional[Dict]:
        """Tłumaczy tekst z opcjonalną korektą błędów i poprawą stylistyki."""
        try:
            prompt = (
                f"Przetłumacz poniższy tekst na język {target_language}."
                + (" Najpierw popraw błędy gramatyczne i stylistyczne, a potem przetłumacz." if correct_errors else "")
                + (" Najpierw popraw stylistykę i płynność tekstu, a potem przetłumacz." if improve_style else "")
                + "\nOdpowiadaj tylko tłumaczeniem, bez wyjaśnień."
                + f"\nTekst: {text}"
            )
            
            messages = [
                {"role": "system", "content": "Jesteś profesjonalnym tłumaczem i nauczycielem języków."},
                {"role": "user", "content": prompt}
            ]
            
            result = self.openai_handler.make_request(messages)
            
            if result:
                return {
                    "translation": result,
                    "original_text": text,
                    "target_language": target_language,
                    "corrected_errors": correct_errors,
                    "improved_style": improve_style
                }
            return None
            
        except Exception as e:
            st.error(f"❌ Błąd podczas tłumaczenia: {str(e)}")
            return None
    
    def generate_audio(self, text: str, language: str = "en", voice: str = "alloy") -> Optional[bytes]:
        """Generuje audio z tekstu używając OpenAI TTS."""
        try:
            # Mapowanie języków na kody TTS
            language_mapping = {
                "Polish": "pl",
                "English": "en", 
                "German": "de",
                "French": "fr",
                "Spanish": "es",
                "Italian": "it",
                "Arabic": "ar",
                "Chinese": "zh",
                "Japanese": "ja"
            }
            
            tts_language = language_mapping.get(language, "en")
            
            # Generowanie audio przez OpenAI TTS
            response = self.openai_handler.client.audio.speech.create(
                model="tts-1",
                voice=voice,  # Można zmienić na: alloy, echo, fable, onyx, nova, shimmer
                input=text,
                response_format="mp3"
            )
            
            # Pobieranie audio
            audio_content = response.content
            
            # Aktualizacja statystyk użycia
            input_tokens = count_tokens(text, "tts-1")
            update_usage_stats(input_tokens, 0, "tts-1")
            
            return audio_content
            
        except Exception as e:
            st.error(f"❌ Błąd podczas generowania audio: {str(e)}")
            return None

# --- FISZKI ---
class FlashcardManager:
    def __init__(self, openai_handler: OpenAIHandler):
        self.openai_handler = openai_handler
        # Inicjalizacja instructor client
        try:
            self.instructor_client = instructor.from_openai(openai_handler.client)
        except Exception as e:
            st.error(f"❌ Błąd inicjalizacji instructor: {e}")
            self.instructor_client = None

    def generate_flashcards(self, text: str, definition_language: str) -> Optional[Dict]:
        """Generuje fiszki używając instructor dla strukturyzowanych odpowiedzi"""
        try:
            if not self.instructor_client:
                st.error("❌ Instructor nie jest dostępny. Używam standardowej metody.")
                return self._generate_flashcards_fallback(text, definition_language)
            
            prompt = (
                f"Wydobądź 4-6 najważniejszych (kluczowych) słów z poniższego tekstu. "
                f"Dla każdego słowa wygeneruj fiszkę z definicją w języku {definition_language} "
                f"i przykładowym zdaniem w oryginalnym języku.\n\n"
                f"Tekst: {text}"
            )
            
            messages = [
                {"role": "system", "content": "Jesteś nauczycielem języków. Generujesz fiszki do nauki słówek."},
                {"role": "user", "content": prompt}
            ]
            
            # Użyj instructor do strukturyzowanej odpowiedzi
            result = self.instructor_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_model=FlashcardSet,
                max_tokens=800
            )
            
            # Konwertuj na format słownika
            flashcards_data = {
                "flashcards": [
                    {
                        "word": card.word,
                        "definition": card.definition,
                        "example": card.example
                    }
                    for card in result.flashcards
                ]
            }
            
            return flashcards_data
            
        except Exception as e:
            st.error(f"❌ Błąd podczas generowania fiszek z instructor: {str(e)}")
            st.info("Używam metody fallback...")
            return self._generate_flashcards_fallback(text, definition_language)

    def _generate_flashcards_fallback(self, text: str, definition_language: str) -> Optional[Dict]:
        """Metoda fallback bez instructor"""
        prompt = (
            f"Wydobądź 4-6 najważniejszych (kluczowych) słów z poniższego tekstu. Nie wybieraj pojedynczych liter ani słów nieistotnych.\n"
            f"Dla każdego słowa wygeneruj fiszkę w formacie JSON:\n"
            f"- word: oryginalne słowo\n"
            f"- definition: krótka definicja w języku {definition_language}\n"
            f"- example: przykładowe zdanie z tym słowem w oryginalnym języku\n"
            "Odpowiadaj TYLKO i wyłącznie w formacie JSON, bez żadnych wyjaśnień, komentarzy, tekstu przed ani po JSON.\n"
            "Przykład odpowiedzi:\n"
            '{"flashcards": ['
            '{"word": "kot", "definition": "domowe zwierzę, często trzymane jako towarzysz", "example": "Mam kota."},'
            '{"word": "pies", "definition": "wierny towarzysz człowieka, często trzymany jako zwierzę domowe", "example": "Pies szczeka w ogrodzie."}'
            ']}\n'
            "Tekst:\n" + text
        )
        
        try:
            messages = [
                {"role": "system", "content": "You are a language teacher. Respond ONLY in JSON format, no extra text."},
                {"role": "user", "content": prompt}
            ]
            result = self.openai_handler.make_request(messages, max_tokens=800)
            
            if not result:
                st.error("❌ OpenAI zwróciło pustą odpowiedź. Spróbuj ponownie lub zmień prompt.")
                return None
                
        except Exception as e:
            st.error(f"❌ Błąd podczas generowania: {str(e)}")
            return None

        try:
            if isinstance(result, dict):
                parsed_result = result
            else:
                cleaned = result.strip()
                # Usuń płotki Markdown i wyciągnij blok JSON
                if cleaned.startswith("```json"):
                    cleaned = cleaned[len("```json"):].strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned[len("```"):].strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
                
                # Jeśli nadal nie zaczyna się od '{', wyciągnij blok JSON regexem
                if not cleaned.startswith("{"):
                    match = re.search(r"\{[\s\S]*\}", cleaned)
                    if match:
                        cleaned = match.group(0)
                
                parsed_result = json.loads(cleaned)
            
            if not isinstance(parsed_result, dict) or "flashcards" not in parsed_result:
                st.error("❌ Odpowiedź nie zawiera klucza 'flashcards'. Odpowiedź:")
                st.code(result)
                return None
            
            return parsed_result
            
        except Exception as e:
            st.error(f"❌ Błąd parsowania JSON: {e}")
            st.info(f"Odpowiedź OpenAI (debug, typ: {type(result)}):")
            st.code(result)
            return None

    def generate_grammar_tips(self, text: str, language: str) -> Optional[Dict]:
        """Generuje wskazówki gramatyczne używając instructor"""
        try:
            if not self.instructor_client:
                st.error("❌ Instructor nie jest dostępny.")
                return None
            
            prompt = (
                f"Przeanalizuj poniższy tekst w języku {language} i wygeneruj 3-5 wskazówek gramatycznych. "
                f"Każda wskazówka powinna zawierać regułę, wyjaśnienie i przykłady.\n\n"
                f"Tekst: {text}"
            )
            
            messages = [
                {"role": "system", "content": f"Jesteś ekspertem od gramatyki języka {language}."},
                {"role": "user", "content": prompt}
            ]
            
            # Użyj instructor do strukturyzowanej odpowiedzi
            result = self.instructor_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_model=GrammarTips,
                max_tokens=600
            )
            
            # Konwertuj na format słownika
            grammar_data = {
                "tips": [
                    {
                        "rule": tip.rule,
                        "explanation": tip.explanation,
                        "examples": tip.examples
                    }
                    for tip in result.tips
                ]
            }
            
            return grammar_data
            
        except Exception as e:
            st.error(f"❌ Błąd podczas generowania wskazówek gramatycznych: {str(e)}")
            return None

    def generate_images(self, flashcards_data: Dict, size_choice: str = "Duże (800×600)", format_choice: str = "PNG (najlepsza jakość)", quality_choice: str = "Wysoka") -> Optional[bytes]:
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
        except ImportError:
            st.error("❌ Brak biblioteki Pillow. Zainstaluj: pip install Pillow")
            return None

        try:
            flashcards = flashcards_data.get("flashcards", [])
            if not flashcards:
                st.error("❌ Brak danych fiszek do wygenerowania obrazów")
                return None

            card_width, card_height = 400, 280
            margin = 25
            font_large_size, font_medium_size, font_small_size = 18, 14, 11

            max_cards = 4
            cards_per_row = 2
            cards_per_col = min(2, (len(flashcards) + 1) // 2)
            total_width = cards_per_row * card_width + (cards_per_row + 1) * margin
            total_height = cards_per_col * card_height + (cards_per_col + 1) * margin + 80

            img = Image.new('RGB', (total_width, total_height), color='white')
            draw = ImageDraw.Draw(img)

            def _load_font_with_fallback(size: int):
                try:
                    return ImageFont.truetype("arial.ttf", size)
                except Exception:
                    return ImageFont.load_default()

            font_large = _load_font_with_fallback(font_large_size)
            font_medium = _load_font_with_fallback(font_medium_size)
            font_small = _load_font_with_fallback(font_small_size)

            title = "📚 Fiszki do nauki"
            title_bbox = draw.textbbox((0, 0), title, font=font_large)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (total_width - title_width) // 2
            draw.text((title_x, 15), title, fill='#1f77b4', font=font_large)

            for i, card in enumerate(flashcards[:max_cards]):
                row = i // cards_per_row
                col = i % cards_per_row
                x = margin + col * (card_width + margin)
                y = 70 + row * (card_height + margin)
                
                # Najpierw rysuj cień (szary)
                shadow_offset = 3
                draw.rectangle([x + shadow_offset, y + shadow_offset, x + card_width + shadow_offset, y + card_height + shadow_offset], 
                             outline='#c0c0c0', width=1, fill='#e0e0e0')
                
                # Potem rysuj białą fiszkę na górze z czarnym obramowaniem
                draw.rectangle([x, y, x + card_width, y + card_height], outline='black', width=2, fill='white')
                # Dodaj niebieskie obramowanie wewnętrzne
                draw.rectangle([x+2, y+2, x + card_width-2, y + card_height-2], outline='#1f77b4', width=1, fill='white')
                
                left_margin = x + 15
                word = card.get("word", "")[:25]
                
                # Słowo - pogrubione i w kolorze
                draw.text((left_margin, y + 15), "SŁOWO:", fill='#1f77b4', font=font_medium)
                draw.text((left_margin + 80, y + 15), word, fill='#2c3e50', font=font_medium)
                
                # Linia oddzielająca
                line_y = y + 45
                draw.line([left_margin, line_y, x + card_width - 15, line_y], fill='#1f77b4', width=2)
                
                definition = card.get("definition", "")[:50]
                def_y = y + card_height//2 + 15
                draw.text((left_margin, def_y), "DEFINICJA:", fill='#1f77b4', font=font_small)
                draw.text((left_margin + 100, def_y), definition, fill='#333', font=font_small)
                
                example = card.get("example", "")[:60]
                ex_y = def_y + 40
                draw.text((left_margin, ex_y), "PRZYKŁAD:", fill='#1f77b4', font=font_small)
                draw.text((left_margin + 100, ex_y), example, fill='#666', font=font_small)

            bio = io.BytesIO()
            try:
                if "JPG" in format_choice:
                    img.save(bio, format='JPEG', quality=85, optimize=True)
                else:
                    img.save(bio, format='PNG', optimize=True)
                bio.seek(0)
                return bio.getvalue()
            except Exception as save_error:
                st.error(f"❌ Błąd podczas zapisywania obrazu: {save_error}")
                return None

        except Exception as e:
            st.error(f"❌ Błąd generowania obrazów: {str(e)}")
            return None

# --- GŁÓWNA APLIKACJA ---
class MultilingualApp:
    def __init__(self):
        self.flashcard_manager = None
        self.client = None
        self.openai_handler = None
        self.translation_manager = None

    def generate_practice_words(self, language: str, practice_type: str, generation_counter: int = 0):
        """Generuje słowa do ćwiczenia wymowy"""
        try:
            # Dodaj różnorodność na podstawie licznika
            variety_instructions = [
                "Używaj prostych, podstawowych słów",
                "Używaj słów średniego poziomu trudności",
                "Używaj bardziej zaawansowanych słów",
                "Używaj słów z różnych dziedzin życia",
                "Używaj słów związanych z podróżowaniem",
                "Używaj słów związanych z jedzeniem",
                "Używaj słów związanych z pracą",
                "Używaj słów związanych z rodziną",
                "Używaj słów związanych z hobby",
                "Używaj słów związanych z technologią"
            ]
            
            variety_instruction = variety_instructions[generation_counter % len(variety_instructions)]
            
            prompts = {
                "Słowa podstawowe": f"Generate 5 very simple, basic words in {language}. Use only simple, everyday words that beginners can easily pronounce. Examples: cat, dog, house, book, car. Format: 1. Word1 2. Word2 3. Word3 4. Word4 5. Word5",
                "Zwroty codzienne": f"Generate 5 simple daily phrases in {language}. {variety_instruction}. Format: 1. Phrase1 2. Phrase2 3. Phrase3 4. Phrase4 5. Phrase5",
                "Liczby": f"Generate numbers 1-10 in {language}. Always use actual numbers like: one, two, three, four, five, six, seven, eight, nine, ten. Format: 1. Number1 2. Number2 3. Number3 4. Number4 5. Number5 6. Number6 7. Number7 8. Number8 9. Number9 10. Number10",
                "Kolory": f"Generate 8 basic colors in {language}. {variety_instruction}. Format: 1. Color1 2. Color2 3. Color3 4. Color4 5. Color5 6. Color6 7. Color7 8. Color8",
                "Członkowie rodziny": f"Generate 8 family members in {language}. Always use family member words like: mother, father, sister, brother, grandmother, grandfather, aunt, uncle. Format: 1. Member1 2. Member2 3. Member3 4. Member4 5. Member5 6. Member6 7. Member7 8. Member8",
            }
            prompt = prompts.get(practice_type, prompts["Słowa podstawowe"])
            messages = [
                {"role": "system", "content": f"Jesteś nauczycielem języka {language}. Generujesz słowa do ćwiczenia wymowy. {variety_instruction}."},
                {"role": "user", "content": prompt},
            ]
            result = self.openai_handler.make_request(messages)
            if result:
                st.success("✅ Słowa do ćwiczeń wygenerowane!")
                # Zapamiętaj wynik
                st.session_state.practice_words_result = result
                st.session_state.practice_words_display_type = practice_type
                st.session_state.practice_words_language = language
            else:
                st.error("❌ Nie udało się wygenerować słów do ćwiczeń.")
        except Exception as e:
            st.error(f"❌ Błąd podczas generowania słów: {e}")

    def analyze_pronunciation(self, language: str, recorded_text: str):
        """Analizuje wymowę na podstawie nagranego tekstu - szybsza wersja"""
        try:
            # Krótszy prompt dla szybszej analizy
            prompt = f"""
            Krótko przeanalizuj wymowę w języku {language}.
            Tekst: "{recorded_text}"
            Format (krótko):
            **Ocena:** X/10
            **Błędy:** [2-3 główne]
            **Wskazówki:** [2-3 konkretne]
            """
            
            messages = [
                {"role": "system", "content": f"Jesteś ekspertem od wymowy języka {language}. Analizujesz wymowę i dajesz konkretne wskazówki."},
                {"role": "user", "content": prompt}
            ]
            
            result = self.openai_handler.make_request(messages, max_tokens=400)
            return result
            
        except Exception as e:
            st.error(f"❌ Błąd podczas analizy wymowy: {str(e)}")
            return None

    def render_translation_section(self):
        st.header("🌍 Tłumaczenie tekstu")
        
        # Ładna ramka dla tłumaczenia
        with st.container():
            st.markdown("""
            <style>
            .translation-box {
                background-color: #f0f2f6;
                border: 2px solid #1f77b4;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            # U góry - wybór języka
            target_lang = st.selectbox(
                "Język docelowy",
                ["English", "German", "French", "Spanish", "Italian"],
                index=0,
                help="Wybierz język na który chcesz przetłumaczyć tekst"
            )
            
            # Pod tym - text area
            text = st.text_area("Wpisz tekst do przetłumaczenia", key="translation_text", height=120, value=st.session_state.get('recorded_audio_text', ''))
            
            # Pod spodem - nagrywanie + opcje poprawy
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Opcje poprawy
                correct_errors = st.checkbox("Popraw błędy gramatyczne", value=False, help="Automatycznie popraw błędy przed tłumaczeniem")
                improve_style = st.checkbox("Popraw stylistykę", value=False, help="Popraw stylistykę i płynność tekstu")
            
            with col2:
                # Nagrywanie audio
                st.markdown("### 🎙️ Nagrywanie")
                
            try:
                from audiorecorder import audiorecorder
                from pydub import AudioSegment  # Upewnij się, że pydub jest zaimportowany

                # Nagrywanie audio - jeden przycisk
                audio_data = audiorecorder(
                    "🔴 Kliknij aby rozpocząć nagrywanie",
                    "⏹️ Kliknij aby zatrzymać",
                    key="pronunciation_voice_recorder"
                )

                if audio_data is not None and len(audio_data) > 0:
                    st.success("✅ **Nagranie zakończone!**")
                    
                    # Konwersja AudioSegment na dane binarne (bytes)
                    try:
                        # Sprawdź format audio przed konwersją
                        st.info(f"🔍 Format audio: {type(audio_data)}")
                        if hasattr(audio_data, 'frame_rate'):
                            st.info(f"🔍 Sample rate: {audio_data.frame_rate} Hz")
                        
                        # Eksportuj do WAV (sekcja tłumaczeń - UNIKALNY)
                        audio_bytes_io = io.BytesIO()
                        
                        # Sprawdź typ audio_data przed eksportem
                        st.info(f"🔍 Typ audio_data: {type(audio_data)}")
                        st.info(f"🔍 Długość audio_data: {len(audio_data)}")
                        
                        # Dodatkowe sprawdzenie obiektu audio
                        if hasattr(audio_data, 'frame_rate'):
                            st.info(f"🔍 Sample rate: {audio_data.frame_rate} Hz")
                        if hasattr(audio_data, 'channels'):
                            st.info(f"🔍 Kanały: {audio_data.channels}")
                        if hasattr(audio_data, 'duration_seconds'):
                            st.info(f"🔍 Czas trwania: {audio_data.duration_seconds:.2f} s")
                        
                        # Spróbuj eksport do różnych formatów
                        try:
                            # Najpierw spróbuj WAV
                            st.info("🔄 Eksportuję do WAV...")
                            audio_data.export(audio_bytes_io, format="wav")
                            audio_bytes_io.seek(0)
                            audio_bytes = audio_bytes_io.getvalue()
                            
                            st.info(f"🔍 Rozmiar po eksporcie WAV: {len(audio_bytes)} bajtów")
                            
                            if len(audio_bytes) >= 4:
                                header = audio_bytes[:4]
                                if header.startswith(b'RIFF'):
                                    st.success("✅ Eksport WAV udany")
                                    # Sprawdź czy plik ma minimalny rozmiar WAV
                                    if len(audio_bytes) < 44:
                                        st.warning("⚠️ Plik WAV jest za mały - może być uszkodzony")
                                else:
                                    st.warning(f"⚠️ Eksport WAV nieudany, nagłówek: {header.hex()}")
                                    # Spróbuj MP3
                                    st.info("🔄 Próbuję MP3...")
                                    audio_bytes_io.seek(0)
                                    audio_bytes_io.truncate(0)
                                    audio_data.export(audio_bytes_io, format="mp3")
                                    audio_bytes_io.seek(0)
                                    audio_bytes = audio_bytes_io.getvalue()
                                    st.info("🔄 Przełączono na format MP3")
                            
                        except Exception as export_error:
                            st.error(f"❌ Błąd eksportu: {str(export_error)}")
                            st.error(f"🔍 Typ błędu eksportu: {type(export_error).__name__}")
                            # Spróbuj MP3 jako fallback
                            try:
                                st.info("🔄 Próbuję MP3 jako fallback...")
                                audio_bytes_io.seek(0)
                                audio_bytes_io.truncate(0)
                                audio_data.export(audio_bytes_io, format="mp3")
                                audio_bytes_io.seek(0)
                                audio_bytes = audio_bytes_io.getvalue()
                                st.info("🔄 Użyto MP3 jako fallback")
                            except Exception as mp3_error:
                                st.error(f"❌ Błąd eksportu MP3: {str(mp3_error)}")
                                return
                        
                        if not audio_bytes:
                            st.error("❌ Błąd konwersji audio")
                            return
                        
                        st.success(f"✅ Audio skonwertowane: {len(audio_bytes)} bajtów")
                        
                        # Debug: sprawdź nagłówek pliku
                        if len(audio_bytes) >= 4:
                            header = audio_bytes[:4]
                            st.info(f"🔍 Nagłówek pliku: {header.hex()}")
                            if header.startswith(b'RIFF'):
                                st.info("✅ Format: WAV")
                            elif header.startswith(b'\xff\xfb') or header.startswith(b'ID3'):
                                st.info("✅ Format: MP3")
                            else:
                                st.warning(f"⚠️ Nieznany format: {header.hex()}")
                        
                        # Wyświetl audio po konwersji
                        st.audio(audio_bytes, format="audio/wav")
                        
                    except Exception as export_error:
                        st.error(f"❌ Błąd eksportu audio: {str(export_error)}")
                        st.error(f"🔍 Typ błędu: {type(export_error).__name__}")
                        return

                    # Transkrypcja audio
                    if st.button("🎧 Transkrybuj nagranie", key="pronunciation_transcribe", type="primary", use_container_width=True):
                        with st.spinner("🎧 Transkrybuję nagranie przez OpenAI Whisper..."):
                            try:
                                transcribed_text = self.openai_handler.transcribe_audio(audio_bytes)
                                if transcribed_text:
                                    st.success("🎧 **Transkrypcja gotowa!**")
                                    st.info(f"📝 **Rozpoznany tekst:** {transcribed_text}")
                                    st.session_state.recorded_audio_text = transcribed_text
                                    st.rerun()  # Odśwież stronę, żeby tekst trafił do text area
                                else:
                                    st.error("❌ Nie udało się przetworzyć audio na tekst.")
                            except Exception as e:
                                st.error(f"❌ Błąd podczas transkrypcji: {str(e)}")
                else:
                    st.warning("⚠️ Brak danych audio do przetworzenia.")
            except ImportError:
                st.error("❌ Brak biblioteki audiorecorder. Zainstaluj: pip install audiorecorder")
            except Exception as e:
                st.error(f"❌ Błąd podczas nagrywania: {str(e)}")
                
            if st.button("Przetłumacz", type="primary", use_container_width=True, key="translate_btn"):
                # Użyj tylko text area jako głównego wejścia
                if not text or not text.strip():
                    st.warning("Wpisz tekst do przetłumaczenia w pole tekstowe.")
                    return
                
                text_to_translate = text
                
                st.session_state.request_count += 1
                
                with st.spinner("Tłumaczę..."):
                    translation_result = self.translation_manager.translate_text(text_to_translate, target_lang, correct_errors, improve_style)
                
                if translation_result:
                    st.success("✅ Tłumaczenie gotowe!")
                    # Wyświetl wynik w ładnej ramce
                    st.markdown(f"""
                    <div class="translation-box">
                        <h4 style="color: #1f77b4; margin-top: 0;">🌍 Tłumaczenie ({target_lang}):</h4>
                        {translation_result["translation"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # TODO: Dodać generowanie audio w przyszłości
                    pass
                    
                else:
                    st.error("❌ Nie udało się przetłumaczyć tekstu.")
                


    def render_explanation_section(self):
        st.header("📚 Wyjaśnienia słów i gramatyki")
        
        # Ładna ramka dla wyjaśnień
        with st.container():
            st.markdown("""
            <style>
            .explanation-box {
                background-color: #f0f2f6;
                border: 2px solid #1f77b4;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            explain_text = st.text_area("Wpisz zdanie lub tekst do wyjaśnienia:", key="explanation_text", height=100)
            
            if st.button("Wyjaśnij", key="explain_btn", type="primary"):
                if not explain_text.strip():
                    st.warning("Wpisz tekst do wyjaśnienia.")
                    return
                
                with st.spinner("Wyjaśniam..."):
                    prompt = (
                        f"Wyjaśnij znaczenie i gramatykę poniższego tekstu w prosty sposób, z przykładami.\nTekst: {explain_text}"
                    )
                    
                    messages = [
                        {"role": "system", "content": "Jesteś nauczycielem języków. Wyjaśniaj prosto, z przykładami."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    result = self.openai_handler.make_request(messages)
                
                if result:
                    st.success("✅ Wyjaśnienie gotowe!")
                    # Wyświetl wynik w ładnej ramce
                    st.markdown(f"""
                    <div class="explanation-box">
                        <h4 style="color: #1f77b4; margin-top: 0;">📖 Wyjaśnienie:</h4>
                        {result}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Nie udało się wygenerować wyjaśnienia.")

    def render_flashcard_section(self):
        st.header("📖 Fiszki ze słówek do nauki")
        
        # Ładna ramka dla fiszek
        with st.container():
            st.markdown("""
            <style>
            .flashcard-box {
                background-color: #f0f2f6;
                border: 2px solid #1f77b4;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.write("📝 Wpisz tekst, z którego chcesz wygenerować fiszki:")
            
            flashcard_text = st.text_area("Tekst do fiszek", key="flashcard_text_new", height=100)
            st.write("Wybierz język definicji fiszek")
            
            definition_lang = st.selectbox(
                "Język definicji", 
                ["Polish", "English", "German", "French", "Spanish", "Italian", "Arabic", "Chinese", "Japanese"], 
                index=0, 
                key="flashcard_def_lang_new"
            )
        
        if st.button("Wygeneruj fiszki", type="primary", use_container_width=True, key="generate_flashcards_new"):
            if not flashcard_text.strip():
                st.warning("Wpisz tekst do wygenerowania fiszek.")
                return
            
            st.session_state.request_count += 1
            
            with st.spinner("Generuję fiszki..."):
                flashcards_data = self.flashcard_manager.generate_flashcards(flashcard_text, definition_lang)
            
            if flashcards_data and "flashcards" in flashcards_data and len(flashcards_data["flashcards"]) > 0:
                st.session_state.flashcards_data = flashcards_data
                st.success("✅ Fiszki zostały wygenerowane!")
                
                st.info("🎨 Automatycznie generuję obraz fiszek...")
                with st.spinner("🎨 Generuję obraz fiszek..."):
                    image_data = self.flashcard_manager.generate_images(flashcards_data)
                
                if image_data:
                    st.session_state.flashcards_image = image_data
                    st.image(image_data, use_container_width=True)
                    
                    # Dodaj timestamp żeby obraz był zawsze świeży
                    import time
                    timestamp = int(time.time())
                    
                    st.download_button(
                        label="📥 Pobierz fiszki jako PNG",
                        data=image_data,
                        file_name=f"flashcards_{timestamp}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                else:
                    st.warning("⚠️ Nie udało się wygenerować obrazu fiszek.")
                
                # Wyświetl fiszki w expanderach
                for i, card in enumerate(st.session_state.flashcards_data["flashcards"], 1):
                    with st.expander(f"🃏 Fiszka {i}: {card.get('word', 'N/A')}"):
                        st.markdown(f"**Definicja:** {card.get('definition', 'N/A')}")
                        st.markdown(f"**Przykład:** {card.get('example', 'N/A')}")
            else:
                st.error("❌ Nie udało się wygenerować fiszek. Sprawdź tekst i spróbuj ponownie.")

    def render_pronunciation_section(self):
        st.header("🎤 Ćwiczenie wymowy")
        st.markdown("---")
        
        # Wybór języka do ćwiczenia
        language = st.selectbox(
            "Język do ćwiczenia wymowy:",
            ["English", "German", "French", "Spanish", "Italian", "Polish"],
            index=0,
            key="pronunciation_language"
        )
        
        # Generowanie słów do ćwiczenia
        practice_type = st.selectbox(
            "Typ ćwiczenia:",
            ["Słowa podstawowe", "Zwroty codzienne", "Liczby", "Kolory", "Członkowie rodziny"],
            index=0,
            key="practice_type"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🎯 Generuj słowa do ćwiczenia", type="primary", use_container_width=True):
                self.generate_practice_words(language, practice_type)
            
            # Wyświetl wygenerowane słowa
            if st.session_state.get('practice_words_result'):
                st.markdown("**📝 Słowa do ćwiczenia:**")
                st.write(st.session_state.practice_words_result)
        
        with col2:
            # Nagrywanie wymowy
            st.markdown("### 🎙️ Nagrywanie wymowy")
            try:
                from audiorecorder import audiorecorder
                audio_data = audiorecorder(
                    "🔴 Kliknij aby rozpocząć nagrywanie",
                    "⏹️ Kliknij aby zatrzymać",
                    key="pronunciation_voice_recorder"
                )
                
                if audio_data is not None and len(audio_data) > 0:
                    st.success("✅ **Nagranie zakończone!**")
                    
                    # Konwersja AudioSegment na bytes w formacie WAV
                    try:
                        # Sprawdź format audio przed konwersją
                        st.info(f"🔍 Format audio: {type(audio_data)}")
                        if hasattr(audio_data, 'frame_rate'):
                            st.info(f"🔍 Sample rate: {audio_data.frame_rate} Hz")
                        
                        # Eksportuj do WAV (sekcja wymowy)
                        audio_bytes_io = io.BytesIO()
                        
                        # Sprawdź typ audio_data przed eksportem
                        st.info(f"🔍 Typ audio_data: {type(audio_data)}")
                        st.info(f"🔍 Długość audio_data: {len(audio_data)}")
                        
                        # Spróbuj eksport do różnych formatów
                        try:
                            # Najpierw spróbuj WAV
                            audio_data.export(audio_bytes_io, format="wav")
                            audio_bytes_io.seek(0)
                            audio_bytes = audio_bytes_io.getvalue()
                            
                            if len(audio_bytes) >= 4:
                                header = audio_bytes[:4]
                                if header.startswith(b'RIFF'):
                                    st.success("✅ Eksport WAV udany")
                                else:
                                    st.warning(f"⚠️ Eksport WAV nieudany, nagłówek: {header.hex()}")
                                    # Spróbuj MP3
                                    audio_bytes_io.seek(0)
                                    audio_bytes_io.truncate(0)
                                    audio_data.export(audio_bytes_io, format="mp3")
                                    audio_bytes_io.seek(0)
                                    audio_bytes = audio_bytes_io.getvalue()
                                    st.info("🔄 Przełączono na format MP3")
                            
                        except Exception as export_error:
                            st.error(f"❌ Błąd eksportu: {str(export_error)}")
                            # Spróbuj MP3 jako fallback
                            try:
                                audio_bytes_io.seek(0)
                                audio_bytes_io.truncate(0)
                                audio_data.export(audio_bytes_io, format="mp3")
                                audio_bytes_io.seek(0)
                                audio_bytes = audio_bytes_io.getvalue()
                                st.info("🔄 Użyto MP3 jako fallback")
                            except Exception as mp3_error:
                                st.error(f"❌ Błąd eksportu MP3: {str(mp3_error)}")
                                return
                        
                        if not audio_bytes:
                            st.error("❌ Błąd konwersji audio")
                            return
                        
                        st.success(f"✅ Audio skonwertowane: {len(audio_bytes)} bajtów")
                        
                        # Debug: sprawdź nagłówek pliku
                        if len(audio_bytes) >= 4:
                            header = audio_bytes[:4]
                            st.info(f"🔍 Nagłówek pliku: {header.hex()}")
                            if header.startswith(b'RIFF'):
                                st.info("✅ Format: WAV")
                            elif header.startswith(b'\xff\xfb') or header.startswith(b'ID3'):
                                st.info("✅ Format: MP3")
                            else:
                                st.warning(f"⚠️ Nieznany format: {header.hex()}")
                        
                        # Wyświetl audio po konwersji
                        st.audio(audio_bytes, format="audio/wav")
                        
                    except Exception as export_error:
                        st.error(f"❌ Błąd eksportu audio: {str(export_error)}")
                        st.error(f"🔍 Typ błędu: {type(export_error).__name__}")
                        return
                    
                    # Transkrypcja audio
                    if st.button("🎧 Transkrybuj nagranie", key="pronunciation_transcribe", type="primary", use_container_width=True):
                        with st.spinner("🎧 Transkrybuję nagranie przez OpenAI Whisper..."):
                            try:
                                # audio_bytes jest już dostępne z poprzedniej konwersji
                                if not audio_bytes:
                                    st.error("❌ Brak danych audio do transkrypcji")
                                    return
                                
                                # Transkrypcja audio
                                transcribed_text = self.openai_handler.transcribe_audio(audio_bytes)
                                
                                if transcribed_text:
                                    st.session_state.practice_text = transcribed_text
                                    st.session_state.recorded_audio_text = transcribed_text
                                    st.session_state.practice_mic_version += 1
                                    st.success("🎧 **Transkrypcja gotowa!**")
                                    st.info(f"📝 **Rozpoznany tekst:** {transcribed_text}")
                                    
                                    # Automatyczna analiza wymowy po nagraniu
                                    st.markdown("**🎯 Analiza wymowy z nagrania**")
                                    analysis_result = self.analyze_pronunciation(language, transcribed_text)
                                    if analysis_result:
                                        st.session_state.last_pronunciation_analysis = analysis_result
                                        st.markdown(f"""
                                        <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 20px 0;">
                                            <h4 style="margin: 0 0 15px 0; color: #1f77b4;">🎤 Analiza wymowy:</h4>
                                            <div style="font-size: 16px; line-height: 1.6; margin: 0; white-space: pre-line;">{analysis_result}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.rerun()  # Odśwież stronę, żeby tekst trafił do text area
                                else:
                                    st.error("❌ Nie udało się przetworzyć audio na tekst.")
                                    
                            except Exception as e:
                                st.error(f"❌ Błąd podczas transkrypcji: {str(e)}")
                                st.error(f"🔍 Typ błędu: {type(e).__name__}")
                
            except ImportError:
                st.error("❌ Brak biblioteki audiorecorder. Zainstaluj: pip install audiorecorder")
            except Exception as e:
                st.error(f"❌ Błąd podczas nagrywania: {str(e)}")
        
        # Analiza wymowy dla tekstu z pola
        st.markdown("---")
        st.markdown("### 📝 Analiza wymowy dla tekstu")
        
        practice_text = st.text_area(
            "Wpisz tekst do analizy wymowy:",
            value=st.session_state.get('practice_text', ''),
            height=100,
            key="practice_text_input"
        )
        
        if st.button("🎯 Analizuj wymowę", type="primary", use_container_width=True):
            if not practice_text.strip():
                st.warning("Wpisz tekst do analizy wymowy.")
                return
            
            with st.spinner("Analizuję wymowę..."):
                analysis_result = self.analyze_pronunciation(language, practice_text)
            
            if analysis_result:
                st.session_state.last_pronunciation_analysis = analysis_result
                st.success("✅ Analiza wymowy gotowa!")
                st.markdown(f"""
                <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 20px 0;">
                    <h4 style="margin: 0 0 15px 0; color: #1f77b4;">🎤 Analiza wymowy:</h4>
                    <div style="font-size: 16px; line-height: 1.6; margin: 0; white-space: pre-line;">{analysis_result}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Nie udało się przeanalizować wymowy.")



    # --- PIERWSZA PLANSZA - KLUCZ API ---
    def render_setup_screen(self):
        """Pierwsza plansza - wprowadzenie klucza API"""
        st.title("🔑 Konfiguracja Language Helper AI")
        st.markdown("---")
        
        st.markdown("""
        <div style="background-color: #f0f2f6; border: 2px solid #1f77b4; border-radius: 10px; padding: 30px; margin: 20px 0; text-align: center;">
            <h2 style="color: #1f77b4; margin-bottom: 20px;">🌍 Witaj w Language Helper AI!</h2>
            <p style="font-size: 18px; margin-bottom: 25px;">Aby rozpocząć korzystanie z aplikacji, musisz wprowadzić swój klucz API OpenAI.</p>
            <p style="color: #666; font-size: 16px;">Klucz zaczyna się od 'sk-' i można go uzyskać na stronie <a href='https://platform.openai.com/api-keys' target='_blank'>OpenAI Platform</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pole do wprowadzenia klucza API
        api_key = st.text_input(
            "Wprowadź swój klucz API OpenAI:",
            type="password",
            placeholder="sk-...",
            help="Twój klucz API OpenAI (zaczyna się od 'sk-')",
            key="api_key_input"
        )
        
        if st.button("🚀 Rozpocznij", type="primary", use_container_width=True):
            if not api_key or not api_key.startswith("sk-"):
                st.error("❌ Podaj poprawny klucz API (zaczyna się od 'sk-').")
            else:
                st.session_state.api_key = api_key
                st.success("✅ Klucz API zaakceptowany!")
                st.rerun()
    
    # --- GŁÓWNA APLIKACJA ---
    def render_main_app(self):
        """Główna aplikacja z wszystkimi funkcjami"""
        # Renderuj sidebar z opcjami (bez klucza API)
        self.render_main_sidebar()
        
        with st.spinner("🔑 Inicjalizuję klienta OpenAI..."):
            self.client = get_openai_client(st.session_state.api_key)
        
        if not self.client:
            st.error("❌ Nie można zainicjalizować klienta OpenAI. Sprawdź klucz API.")
            return
        
        self.openai_handler = OpenAIHandler(self.client)
        self.translation_manager = TranslationManager(self.openai_handler)
        self.flashcard_manager = FlashcardManager(self.openai_handler)

        st.title("🌍 Language Helper AI")
        st.markdown("---")
        
        self.render_translation_section()
        st.markdown("---")
        
        self.render_explanation_section()
        st.markdown("---")
        
        self.render_flashcard_section()
        st.markdown("---")
        
        # 📚 Wskazówki gramatyczne
        self.render_grammar_tips_section()
        st.markdown("---")
        
        # 🎤 ĆWICZENIE WYMOWY - GŁÓWNA SEKCJA
        st.header("🎤 Ćwiczenie wymowy")
        st.markdown("---")
        
        self.render_pronunciation_section()
        
        # Renderuj stopkę
        self.render_footer()
    
    def render_main_sidebar(self):
        """Sidebar głównej aplikacji (bez klucza API)"""
        st.sidebar.title("🌍 Language Helper AI")
        st.sidebar.markdown("---")
        
        # Motyw
        st.sidebar.markdown("### 🎨 Motyw")
        theme = st.sidebar.radio("Wybierz motyw", ["Jasny", "Ciemny"], index=0)
        if theme == "Ciemny":
            st.markdown("""
            <style>
            body, .stApp {background-color: #0e1117 !important; color: #fafafa !important;}
            </style>
            """, unsafe_allow_html=True)
        
        # Statystyki
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Statystyki API")
        st.sidebar.metric("Tokeny", f"{st.session_state.total_tokens:,}")
        st.sidebar.metric("Koszt", f"${st.session_state.total_cost:.4f}")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Aplikacja:**")
        st.sidebar.markdown("🌍 tłumaczenia")
        st.sidebar.markdown("📖 fiszki")
        st.sidebar.markdown("📚 wyjaśnienia")
        st.sidebar.markdown("📚 wskazówki gramatyczne")
        st.sidebar.markdown("🎤 wymowa")
        st.sidebar.markdown("🎙️ nagrywanie audio")
        
        # Opcja resetowania klucza API
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔑 Zarządzanie API")
        if st.sidebar.button("🔄 Zmień klucz API", type="secondary", use_container_width=True):
            st.session_state.api_key = ""
            st.rerun()
    
    def render_grammar_tips_section(self):
        """Sekcja wskazówek gramatycznych"""
        st.header("📚 Wskazówki gramatyczne")
        st.markdown("---")
        
        # Ładna ramka dla wskazówek gramatycznych
        with st.container():
            st.markdown("""
            <style>
            .grammar-box {
                background-color: #f0f2f6;
                border: 2px solid #1f77b4;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            grammar_text = st.text_area("Wpisz tekst do analizy gramatycznej:", key="grammar_text", height=100)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                grammar_language = st.selectbox(
                    "Język tekstu:",
                    ["Polish", "English", "German", "French", "Spanish", "Italian"],
                    index=0,
                    key="grammar_language"
                )
            
            with col2:
                if st.button("📚 Generuj wskazówki gramatyczne", type="primary", use_container_width=True):
                    if not grammar_text.strip():
                        st.warning("Wpisz tekst do analizy gramatycznej.")
                        return
                    
                    st.session_state.request_count += 1
                    
                    with st.spinner("Generuję wskazówki gramatyczne..."):
                        grammar_result = self.flashcard_manager.generate_grammar_tips(grammar_text, grammar_language)
                    
                    if grammar_result and "tips" in grammar_result:
                        st.success("✅ Wskazówki gramatyczne gotowe!")
                        
                        for i, tip in enumerate(grammar_result["tips"], 1):
                            with st.expander(f"📖 Wskazówka {i}: {tip.get('rule', 'N/A')}"):
                                st.markdown(f"**Reguła:** {tip.get('rule', 'N/A')}")
                                st.markdown(f"**Wyjaśnienie:** {tip.get('explanation', 'N/A')}")
                                st.markdown("**Przykłady:**")
                                for example in tip.get('examples', []):
                                    st.markdown(f"- {example}")
                    else:
                        st.error("❌ Nie udało się wygenerować wskazówek gramatycznych.")
    
    def render_footer(self):
        """Stopka aplikacji"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>🌍 <strong>Language Helper AI</strong> - Twój inteligentny asystent do nauki języków</p>
            <p>Powered by OpenAI GPT-4 & Whisper</p>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Główna metoda uruchamiająca aplikację"""
        # Sprawdź czy klucz API jest wprowadzony
        if not st.session_state.api_key or not st.session_state.api_key.startswith("sk-"):
            # PIERWSZA PLANSZA - wprowadzenie klucza API
            self.render_setup_screen()
        else:
            # GŁÓWNA APLIKACJA - renderuj sidebar i wszystkie funkcje
            self.render_main_app()

# --- URUCHOMIENIE ---
if __name__ == "__main__":
    # Inicjalizacja session state
    init_session_state()
    
    # Inicjalizacja aplikacji
    app = MultilingualApp()
    app.run()