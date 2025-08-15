import streamlit as st
from openai import OpenAI
import json
from datetime import datetime
from typing import Dict, List, Optional
import tiktoken
import io
import re
import speech_recognition as sr

# --- USTAWIENIA STRONY ---
st.set_page_config(
    page_title="Language Helper AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- INICJALIZACJA SESJI ---
def init_session_state():
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
    # TODO: Dodać dodatkowe zmienne session state w przyszłości
    pass

# --- SIDEBAR & SETUP ---
def render_sidebar_and_setup():
    st.sidebar.title("🌍 Language Helper AI")
    st.sidebar.markdown("---")
    
    # Klucz API
    st.sidebar.markdown("### 🔑 Klucz API OpenAI")
    api_key = st.sidebar.text_input(
        "Wprowadź swój klucz API OpenAI:",
        type="password",
        placeholder="sk-...",
        value=st.session_state.api_key,
        help="Twój klucz API OpenAI (zaczyna się od 'sk-')"
    )
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    if not api_key or not api_key.startswith("sk-"):
        st.sidebar.warning("Podaj poprawny klucz API (zaczyna się od 'sk-').")
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
    st.sidebar.markdown("🎤 wymowa")

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
            
            # Transkrypcja audio
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=io.BytesIO(file_bytes),
                language=whisper_language,
                response_format="text"
            )
            
            # Aktualizacja statystyk użycia (Whisper nie zwraca tokenów)
            update_usage_stats(0, 0, "whisper-1")
            
            return response
            
        except Exception as e:
            st.error(f"❌ Błąd podczas transkrypcji audio: {str(e)}")
            return None

# --- SPEECH RECOGNITION MANAGER ---
class SpeechRecognitionManager:
    """Zarządzanie rozpoznawaniem mowy"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def get_audio_from_microphone(self) -> Optional[str]:
        """Nagrywanie audio z mikrofonu i konwersja na tekst"""
        try:
            # Sprawdź dostępność mikrofonu
            try:
                microphone = sr.Microphone()
            except Exception as mic_error:
                raise Exception(f"Błąd dostępu do mikrofonu: {mic_error}. Sprawdź czy mikrofon jest podłączony i działa.")
            
            # Użyj mikrofonu
            with microphone as source:
                # Dostosuj do hałasu otoczenia
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
                # Ustaw parametry dla szybszego nagrywania
                self.recognizer.energy_threshold = 200  # Niższy próg energii = szybsze wykrycie
                self.recognizer.dynamic_energy_threshold = False  # Wyłącz dynamiczny próg
                self.recognizer.pause_threshold = 0.8  # Krótszy próg pauzy = szybsze zatrzymanie
                self.recognizer.non_speaking_duration = 0.8
                
                # Nagrywaj audio z szybszymi parametrami
                audio = self.recognizer.listen(
                    source, 
                    timeout=5,  # Krótszy timeout na rozpoczęcie mówienia
                    phrase_time_limit=15  # Krótszy limit na frazę
                )
                
            # Konwertuj audio na tekst z lepszym rozpoznawaniem
            try:
                # Pierwsza próba - standardowe rozpoznawanie
                text = self.recognizer.recognize_google(audio, language='pl-PL')
                if text:
                    return text
                
                # Druga próba - bez określania języka (automatyczne wykrywanie)
                text = self.recognizer.recognize_google(audio)
                if text:
                    return text
                
                # Trzecia próba - z angielskim jako fallback
                text = self.recognizer.recognize_google(audio, language='en-US')
                if text:
                    return text
                
                return None
                
            except sr.UnknownValueError:
                # Jeśli nie rozpoznano mowy, spróbuj ponownie z innymi ustawieniami
                try:
                    # Zmień parametry i spróbuj ponownie
                    self.recognizer.energy_threshold = 150
                    self.recognizer.pause_threshold = 1.0
                    
                    # Ponowna próba nagrania
                    audio_retry = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
                    text = self.recognizer.recognize_google(audio, language='pl-PL')
                    
                    if text:
                        return text
                    else:
                        return None
                        
                except Exception:
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
                text = self.recognizer.recognize_google(source, language='pl-PL')
                
            if text:
                return text
            else:
                return None
                    
        except Exception as e:
            return None

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

    def generate_flashcards(self, text: str, definition_language: str) -> Optional[Dict]:
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
                # Przycisk nagrywania audio
                st.markdown("### 🎙️ Nagrywanie")
                
                if st.button("🔴 Rozpocznij nagrywanie", key="translation_start_recording", type="secondary", use_container_width=True):
                    try:
                        # Inicjalizacja SpeechRecognitionManager
                        if not hasattr(self, 'translation_speech_manager'):
                            self.translation_speech_manager = SpeechRecognitionManager()
                        
                        with st.spinner("🎙️ Nagrywam... Mów do mikrofonu!"):
                            recorded_text = self.translation_speech_manager.get_audio_from_microphone()
                            
                            if recorded_text:
                                st.session_state.recorded_audio_text = recorded_text
                                st.success("✅ Nagranie zakończone!")
                                st.info(f"🎤 Rozpoznany tekst: **{recorded_text}**")
                                st.rerun()  # Odśwież stronę, aby text area się zaktualizowała
                            else:
                                st.error("❌ Nie udało się rozpoznać mowy. Spróbuj ponownie.")
                                
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
        
        # Ładna ramka dla wymowy
        with st.container():
            st.markdown("""
            <style>
            .pronunciation-box {
                background-color: #f0f2f6;
                border: 2px solid #1f77b4;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.write("Nagraj swoją wymowę, aby przeanalizować poprawność:")
            
            # Wybór języka
            language = st.selectbox(
                "Język", 
                ["Polish", "English", "German", "French", "Spanish", "Italian", "Arabic", "Chinese", "Japanese"], 
                index=0, 
                key="pronunciation_lang"
            )
            
            # 📚 Generuj słowa do ćwiczeń
            st.markdown("**📚 Generuj słowa do ćwiczeń**")
            
            practice_type = st.selectbox(
                "Typ ćwiczeń",
                ["Słowa podstawowe", "Zwroty codzienne", "Liczby", "Kolory", "Członkowie rodziny"],
                key="practice_type"
            )
            
            if st.button("🎯 Generuj słowa do ćwiczeń", type="primary", use_container_width=True):
                # Zwiększ licznik żeby wygenerować inne słówka
                if 'word_generation_counter' not in st.session_state:
                    st.session_state.word_generation_counter = 0
                st.session_state.word_generation_counter += 1
                self.generate_practice_words(language, practice_type, st.session_state.word_generation_counter)
            
            # Wyświetl słowa do ćwiczeń jeśli są wygenerowane (zawsze widoczne)
            if hasattr(st.session_state, 'practice_words_result') and st.session_state.practice_words_result:
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #6f42c1; margin: 20px 0;">
                    <h4 style="margin: 0 0 15px 0; color: #6f42c1;">📚 {st.session_state.practice_words_display_type} ({st.session_state.practice_words_language}):</h4>
                    <div style="font-size: 16px; line-height: 1.0; margin: 0; white-space: pre-line; padding: 5px 0;">{st.session_state.practice_words_result}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🔴 Rozpocznij nagrywanie", key="start_recording", type="secondary", use_container_width=True):
                try:
                    # Inicjalizacja SpeechRecognitionManager
                    if not hasattr(self, 'speech_manager'):
                        self.speech_manager = SpeechRecognitionManager()
                    
                    # Inicjalizacja SpeechRecognitionManager
                    if not hasattr(self, 'speech_manager'):
                        self.speech_manager = SpeechRecognitionManager()
                    
                    # Pokaż informację o rozpoczęciu nagrywania
                    st.info("🎙️ Rozpoczynam nagrywanie... Mów do mikrofonu!")
                    
                    # Nagrywanie z timeout
                    recorded_text = self.speech_manager.get_audio_from_microphone()
                    
                    if recorded_text:
                        # Zapisuj tekst w practice_text
                        st.session_state.practice_text = recorded_text
                        # Zwiększ licznik wersji mikrofonu
                        st.session_state.practice_mic_version += 1
                        st.success("✅ Nagranie zakończone!")
                        st.info(f"🎤 Rozpoznany tekst: **{recorded_text}**")
                        
                        # Automatyczna analiza wymowy po nagraniu
                        st.markdown("**🎯 Analiza wymowy z nagrania**")
                        # Zapisz analizę w session state
                        analysis_result = self.analyze_pronunciation(language, recorded_text)
                        if analysis_result:
                            st.session_state.last_pronunciation_analysis = analysis_result
                            # Wyświetl analizę
                            st.markdown(f"""
                            <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 20px 0;">
                                <h4 style="margin: 0 0 15px 0; color: #1f77b4;">🎤 Analiza wymowy:</h4>
                                <div style="font-size: 16px; line-height: 1.6; margin: 0; white-space: pre-line;">{analysis_result}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Nie udało się rozpoznać mowy. Spróbuj ponownie.")
                            
                except Exception as e:
                    st.error(f"❌ Błąd podczas nagrywania: {str(e)}")
            

            


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
                "Słowa podstawowe": f"Generate 5 basic words in {language}. {variety_instruction}. Format: 1. Word1 2. Word2 3. Word3 4. Word4 5. Word5",
                "Zwroty codzienne": f"Generate 5 common daily phrases in {language}. {variety_instruction}. Format: 1. Phrase1 2. Phrase2 3. Phrase3 4. Phrase4 5. Phrase5",
                "Liczby": f"Generate numbers 1-10 in {language}. {variety_instruction}. Format: 1. Number1 2. Number2 3. Number3 4. Number4 5. Number5 6. Number6 7. Number7 8. Number8 9. Number9 10. Number10",
                "Kolory": f"Generate 8 basic colors in {language}. {variety_instruction}. Format: 1. Color1 2. Color2 3. Color3 4. Color4 5. Color5 6. Color6 7. Color7 8. Color8",
                "Członkowie rodziny": f"Generate 8 family members in {language}. {variety_instruction}. Format: 1. Member1 2. Member2 3. Member3 4. Member4 5. Member5 6. Member6 7. Member7 8. Member8",
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
            **Ćwiczenia:** [2-3 ćwiczenia]
            """
            messages = [
                {"role": "system", "content": f"Jesteś ekspertem od wymowy języka {language}. Odpowiadaj krótko i konkretnie."},
                {"role": "user", "content": prompt},
            ]
            
            # Dodaj timeout do request
            with st.spinner("🔍 Szybka analiza wymowy..."):
                result = self.openai_handler.make_request(messages)
                
            if result:
                return result
            else:
                st.error("❌ Nie udało się przeanalizować wymowy.")
                return None
        except Exception as e:
            st.error(f"❌ Błąd podczas analizy wymowy: {e}")
            return None

    def _analyze_pronunciation_from_transcription(self, transcription_text: str, language: str):
        """Analizuje wymowę na podstawie transkrypcji tekstu"""
        if not transcription_text.strip():
            st.warning("Brak tekstu do analizy.")
            return
        
        st.session_state.request_count += 1
        
        with st.spinner("🔍 Szybka analiza wymowy..."):
            prompt = (
                f"Krótko przeanalizuj wymowę w języku {language}.\n"
                f"Tekst: \"{transcription_text}\"\n"
                "Format (krótko):\n"
                "**Ocena:** X/10\n"
                "**Błędy:** [2-3 główne]\n"
                "**Wskazówki:** [2-3 konkretne]\n"
                "**Ćwiczenia:** [2-3 ćwiczenia]"
            )
            
            messages = [
                {"role": "system", "content": f"Jesteś ekspertem od wymowy języka {language}."},
                {"role": "user", "content": prompt},
            ]
            
            result = self.openai_handler.make_request(messages)
            
            if result:
                st.success("✅ Analiza wymowy gotowa!")
                # Wyświetl wynik w ładnej ramce
                st.markdown(f"""
                <div class="pronunciation-box">
                    <h4 style="color: #1f77b4; margin-top: 0;">🎤 Analiza wymowy:</h4>
                    {result}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Nie udało się przeanalizować wymowy.")

    def _analyze_pronunciation_audio(self, audio_data, language: str):
        """Analizuje wymowę na podstawie nagrania audio"""
        st.session_state.request_count += 1
        
        with st.spinner("🔍 Szybka analiza wymowy..."):
            prompt = (
                f"Krótko przeanalizuj wymowę w języku {language}.\n"
                f"Użytkownik nagrał audio.\n"
                "Format (krótko):\n"
                "**Ocena:** X/10\n"
                "**Błędy:** [2-3 główne]\n"
                "**Wskazówki:** [2-3 konkretne]\n"
                "**Ćwiczenia:** [2-3 ćwiczenia]"
            )
            
            messages = [
                {"role": "system", "content": f"Jesteś ekspertem od wymowy języka {language}."},
                {"role": "user", "content": prompt},
            ]
            
            result = self.openai_handler.make_request(messages)
            
            if result:
                st.success("✅ Analiza wymowy gotowa!")
                # Wyświetl wynik w ładnej ramce
                st.markdown(f"""
                <div class="pronunciation-box">
                    <h4 style="color: #1f77b4; margin-top: 0;">🎤 Analiza wymowy:</h4>
                    {result}
                </div>
                """, unsafe_allow_html=True)
                
                # TODO: Dodać generowanie audio z przykładów wymowy w przyszłości
                pass
            else:
                st.error("❌ Nie udało się przeanalizować wymowy.")

    def render_footer(self):
        """Renderowanie stopki"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>🌍 Language Helper AI - Twoje narzędzie do nauki języków</p>
            <p>Stworzone z ❤️ w Python i Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

    def run(self):
        render_sidebar_and_setup()
        
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
        
        # 🎤 ĆWICZENIE WYMOWY - GŁÓWNA SEKCJA
        st.header("🎤 Ćwiczenie wymowy")
        st.markdown("---")
        
        self.render_pronunciation_section()
        
        # Renderuj stopkę
        self.render_footer()

# --- URUCHOMIENIE ---
if __name__ == "__main__":
    try:
        init_session_state()
        app = MultilingualApp()
        app.run()
    except Exception as e:
        st.error("🚨 **Krytyczny błąd aplikacji!**")
        st.error(f"**Błąd:** {str(e)}")
        st.error("**Typ błędu:** " + type(e).__name__)
