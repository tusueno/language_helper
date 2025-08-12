# 🌍 Tłumacz Wielojęzyczny

Zaawansowana aplikacja do tłumaczenia i nauki języków obcych, zbudowana w Streamlit z wykorzystaniem OpenAI API. **Użytkownicy używają własnych kluczy API i widzą swoje koszty w czasie rzeczywistym.**

## 🚀 **NOWOŚĆ: Własne klucze API i śledzenie kosztów!**
- 🔑 **Każdy użytkownik wprowadza swój klucz API**
- 💰 **Automatyczne liczenie tokenów i kosztów**
- 📊 **Statystyki użycia w czasie rzeczywistym**
- 🔒 **Bezpieczeństwo - klucze nie są zapisywane na serwerze**

## ✨ Funkcjonalności

### 🚀 Tłumaczenie wielojęzyczne
- Tłumaczenie tekstów na 13 języków
- Automatyczna transkrypcja fonetyczna
- Generowanie audio z tłumaczeń
- Cache'owanie wyników dla szybszego działania

### 📚 Wyjaśnienia gramatyczne
- Analiza trudniejszych słów
- Wyjaśnienia konstrukcji gramatycznych
- Prosty język dostosowany do nauki

### ✨ Poprawa stylistyki
- Korekta błędów gramatycznych
- Poprawa płynności wypowiedzi
- Generowanie naturalnych wersji tekstu

### 🔧 Korekcja i tłumaczenie
- Automatyczne wykrywanie języka
- Poprawa błędów w tekście obcojęzycznym
- Tłumaczenie na język polski

### 📖 Fiszki do nauki
- Generowanie fiszek ze słówek
- Definicje i przykłady użycia
- Pobieranie w formacie .txt

### 🎤 Rozpoznawanie mowy (NOWOŚĆ!)
- **Nagrywanie z mikrofonu** - mów zamiast pisać!
- **Wczytywanie plików audio** - WAV, MP3
- **Automatyczna konwersja mowy na tekst**
- **Wsparcie dla wszystkich sekcji aplikacji**

### 🎯 Ćwiczenie wymowy
- **Generowanie słów do ćwiczenia** - podstawowe słowa, zwroty, liczby, kolory
- **Nagrywanie własnej wymowy** - ćwiczenie wymowy
- **Analiza wymowy przez AI** - ocena, wskazówki, ćwiczenia
- **Wsparcie dla 11 języków** - od polskiego po koreański

## 🛠️ Instalacja

### Wymagania
- Python 3.8+
- OpenAI API key (każdy użytkownik wprowadza własny)
- **Mikrofon** (dla funkcji rozpoznawania mowy)
- **Dostęp do internetu** (dla Google Speech Recognition API)

### Kroki instalacji

1. **Sklonuj repozytorium**
```bash
git clone <repository-url>
cd projekt_app
```

2. **Zainstaluj zależności**
```bash
pip install -r requirements.txt
```

3. **Uruchom aplikację lokalnie**
```bash
streamlit run streamlit_app.py
```

4. **Wprowadź swój klucz API**
   - Zarejestruj się na [platform.openai.com](https://platform.openai.com)
   - Wygeneruj klucz API w sekcji "API Keys"
   - Wklej klucz w aplikacji (sidebar)

### 🚀 Wdrożenie na Streamlit Cloud

1. **Przygotuj repozytorium Git**
```bash
git add .
git commit -m "Initial commit: Multilingual Translator App"
git push origin main
```

2. **Wdróż na Streamlit Cloud**
   - Wejdź na [share.streamlit.io](https://share.streamlit.io)
   - Połącz swoje repozytorium GitHub
   - Ustaw główny plik jako `streamlit_app.py`
   - Kliknij "Deploy"

3. **Użytkownicy mogą teraz:**
   - Uruchamiać aplikację bez instalacji
   - Wprowadzać własne klucze API
   - Widzieć swoje koszty w czasie rzeczywistym

## 🎨 Funkcje interfejsu

### 🌐 Wielojęzyczny interfejs
- Polski, Angielski, Niemiecki, Ukraiński
- Francuski, Hiszpański, Arabski
- Chiński, Japoński

### 🎨 Motywy
- Jasny motyw (domyślny)
- Ciemny motyw
- Responsywny design

### 📱 Funkcje UX
- **Rozpoznawanie mowy** - mów zamiast pisać w każdej sekcji
- **Ćwiczenie wymowy** - dedykowana sekcja w sidebarze
- **Analiza wymowy przez AI** - ocena i wskazówki
- Wskaźniki ładowania
- Placeholdery w polach tekstowych
- Komunikaty błędów i sukcesu
- Statystyki użycia

## 🔧 Architektura

### Klasy główne
- `MultilingualApp` - główna aplikacja
- `OpenAIHandler` - obsługa API OpenAI
- `TranslationManager` - zarządzanie tłumaczeniami
- `ExplanationManager` - zarządzanie wyjaśnieniami
- `StyleManager` - zarządzanie stylistyką
- `CorrectionManager` - zarządzanie korekcją
- `FlashcardManager` - zarządzanie fiszkami

### Funkcje pomocnicze
- Cache'owanie wyników
- Walidacja danych wejściowych
- Rate limiting
- Obsługa błędów

## ⚙️ Konfiguracja

### Zmienne środowiskowe
- `MAX_TEXT_LENGTH` - maksymalna długość tekstu (domyślnie 5000)
- `CACHE_TTL` - czas życia cache w sekundach (domyślnie 3600)
- `RATE_LIMIT_DELAY` - opóźnienie między requestami (domyślnie 1s)

**Uwaga:** Klucze API są wprowadzane przez użytkowników w interfejsie aplikacji.

### Ustawienia aplikacji
- Layout: wide
- Sidebar: expanded
- Motyw: jasny/ciemny
- Język interfejsu: 9 opcji

## 🚀 Użycie

1. **Tłumaczenie**
   - Wybierz język docelowy
   - Wpisz tekst do przetłumaczenia
   - Kliknij "Przetłumacz"
   - Odsłuchaj audio (opcjonalnie)

2. **Wyjaśnienia**
   - Wpisz tekst do wyjaśnienia
   - Kliknij "Wyjaśnij słowa i gramatykę"

3. **Poprawa stylistyki**
   - Wpisz tekst do poprawy
   - Kliknij "Popraw stylistykę"

4. **Korekcja błędów**
   - Wpisz tekst w języku obcym
   - Kliknij "Wykryj język i popraw błędy"

5. **Fiszki**
   - Wpisz tekst źródłowy
   - Kliknij "Wygeneruj fiszki"
   - Pobierz plik .txt

6. **Rozpoznawanie mowy**
   - W każdej sekcji kliknij "🎤 Nagryj z mikrofonu"
   - Mów wyraźnie do mikrofonu
   - Tekst automatycznie pojawi się w polu tekstowym

7. **Ćwiczenie wymowy**
   - W sidebarze wybierz "🎤 Ćwicz wymowę"
   - Wybierz język i typ ćwiczenia
   - Kliknij "🎲 Generuj słowa do ćwiczenia"
   - Nagryj swoją wymowę i przeanalizuj ją

## 🔒 Bezpieczeństwo

- **Własne klucze API** - każdy użytkownik używa swojego klucza
- **Lokalne przechowywanie** - klucze nie są zapisywane na serwerze
- **Sesja przeglądarki** - klucze są przechowywane tylko w sesji użytkownika
- Walidacja długości tekstu
- Rate limiting dla API
- Cache'owanie z TTL
- Obsługa błędów API
- Logowanie błędów

## 📊 Wydajność i koszty

- Cache'owanie wyników (1 godzina)
- Rate limiting (1 sekunda między requestami)
- Asynchroniczne przetwarzanie
- Optymalizacja promptów

### 💰 Koszty API (aktualne ceny OpenAI)
- **GPT-4o:** $0.005/1k tokenów wejściowych, $0.015/1k tokenów wyjściowych
- **GPT-4o-mini:** $0.00015/1k tokenów wejściowych, $0.0006/1k tokenów wyjściowych
- **TTS (audio):** $0.015/1k znaków

**Aplikacja automatycznie liczy tokeny i koszty dla każdego requestu.**

## 🐛 Rozwiązywanie problemów

### Błąd inicjalizacji OpenAI
- Sprawdź czy wprowadziłeś poprawny klucz API w sidebarze
- Upewnij się że klucz API jest aktywny i ma wystarczające środki
- Sprawdź format klucza (powinien zaczynać się od 'sk-')

### Błąd generowania audio
- Sprawdź połączenie internetowe
- Upewnij się że API key ma dostęp do TTS

### Problemy z cache
- Sprawdź uprawnienia do zapisu
- Zrestartuj aplikację

### Problemy z rozpoznawaniem mowy
- **Mikrofon nie działa**: Sprawdź uprawnienia do mikrofonu w przeglądarce
- **Błąd "No module named 'pyaudio'"**: Zainstaluj `pip install pyaudio`
- **Błąd rozpoznawania**: Upewnij się że mówisz wyraźnie i w miarę cichym otoczeniu
- **Błąd "speech_recognition"**: Zainstaluj `pip install SpeechRecognition`
- **Problemy z plikami audio**: Używaj formatów WAV lub MP3

## 🚀 Wdrożenie i współpraca

### 📱 Streamlit Cloud
Aplikacja jest gotowa do wdrożenia na Streamlit Cloud:
- Używa `streamlit_app.py` jako głównego pliku
- Konfiguracja w `.streamlit/config.toml`
- Użytkownicy wprowadzają własne klucze API
- Automatyczne śledzenie kosztów

### 🤝 Współpraca
Zachęcamy do:
- Zgłaszania błędów
- Proponowania nowych funkcji
- Poprawiania kodu
- Dodawania nowych języków
- Ulepszania systemu śledzenia kosztów

## 📄 Licencja

Projekt jest dostępny na licencji MIT.

## 🙏 Podziękowania

- OpenAI za API
- Streamlit za framework
- Społeczność open source

---

**Made with ❤️ using Streamlit & OpenAI**
