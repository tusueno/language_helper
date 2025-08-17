# 🌍 Language Helper AI

Inteligentna aplikacja do nauki języków wykorzystująca OpenAI GPT-4 i inne modele AI.

## 🚀 Funkcje

- **🌍 Tłumaczenia** - tłumaczenie tekstu z korektą błędów i poprawą stylistyki
- **📖 Fiszki** - automatyczne generowanie fiszek ze słówek używając instructor
- **📚 Wyjaśnienia** - szczegółowe wyjaśnienia gramatyki i znaczenia słów
- **📚 Wskazówki gramatyczne** - strukturyzowane wskazówki gramatyczne z instructor
- **🎤 Ćwiczenie wymowy** - analiza wymowy z nagrań audio
- **🎙️ Nagrywanie audio** - nagrywanie i transkrypcja audio używając audiorecorder

## 📋 Wymagania

- Python 3.8+
- OpenAI API key
- Mikrofon (dla funkcji nagrywania)

## 🛠️ Instalacja

1. **Sklonuj repozytorium:**
```bash
git clone <repository-url>
cd projekt_app
```

2. **Zainstaluj zależności:**
```bash
pip install -r requirements.txt
```

3. **Skonfiguruj zmienne środowiskowe:**
   
   Utwórz plik `.env` w katalogu `projekt_app`:
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```

   Lub ustaw zmienną środowiskową:
   ```bash
   # Windows
   set OPENAI_API_KEY=sk-your-openai-api-key-here
   
   # Linux/Mac
   export OPENAI_API_KEY=sk-your-openai-api-key-here
   ```

## 🚀 Uruchomienie

```bash
streamlit run app.py
```

Aplikacja będzie dostępna pod adresem: http://localhost:8501

## 🔧 Użycie

1. **Wprowadź klucz API OpenAI** w sidebar (lub ustaw w zmiennych środowiskowych)
2. **Wybierz funkcję** z menu głównego
3. **Użyj funkcji nagrywania** do wprowadzania tekstu przez mikrofon
4. **Generuj fiszki i wskazówki** używając instructor

## 📚 Zależności

- `streamlit` - interfejs użytkownika
- `openai` - integracja z OpenAI API (GPT-4, Whisper, TTS)
- `audiorecorder` - nagrywanie audio z mikrofonu
- `instructor` - strukturyzowane odpowiedzi AI
- `pydantic` - walidacja danych
- `python-dotenv` - zarządzanie zmiennymi środowiskowymi
- `tiktoken` - liczenie tokenów
- `Pillow` - generowanie obrazów fiszek

## 🎯 Funkcje AI

- **GPT-4** - tłumaczenia, wyjaśnienia, analiza wymowy
- **Whisper** - transkrypcja audio do tekstu
- **TTS** - generowanie audio z tekstu
- **Instructor** - strukturyzowane fiszki i wskazówki gramatyczne

## 🔒 Bezpieczeństwo

- Klucz API jest przechowywany lokalnie
- Wszystkie żądania są szyfrowane
- Brak logowania danych użytkownika

## 📝 Licencja

MIT License
