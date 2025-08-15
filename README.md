# 🌍 Language Helper AI

Inteligentna aplikacja do nauki języków obcych zintegrowana z OpenAI API, zbudowana w Python i Streamlit.

## ✨ Funkcje

### 🔤 Tłumaczenie tekstu
- **Tłumaczenie wielojęzyczne** (EN, DE, FR, ES, IT → PL)
- **Nagrywanie audio** z mikrofonu do tekstu
- **Poprawa błędów gramatycznych** i stylistycznych
- **Automatyczne tłumaczenie** nagranego audio

### 📚 Generowanie fiszek
- **Inteligentne fiszki** na podstawie tekstu
- **Obrazy fiszek** generowane przez DALL-E
- **Pobieranie jako PNG** do wydruku
- **Polskie nagłówki** (Słowo, Definicja, Przykład)

### 🎤 Ćwiczenie wymowy
- **Generowanie słówek** do ćwiczeń
- **Nagrywanie audio** z mikrofonu
- **Automatyczna analiza wymowy** przez AI
- **Różnorodne kategorie** słówek (podstawowe, codzienne, kolory, liczby)

### 📖 Wyjaśnienia gramatyczne
- **Inteligentne wyjaśnienia** reguł gramatycznych
- **Przykłady użycia** w kontekście
- **Personalizowane wyjaśnienia** na podstawie tekstu

## 🚀 Instalacja

1. **Sklonuj repozytorium:**
```bash
git clone [URL_REPOZYTORIUM]
cd projekt_app
```

2. **Zainstaluj zależności:**
```bash
pip install -r requirements.txt
```

3. **Ustaw API key OpenAI:**
```bash
export OPENAI_API_KEY="twój_klucz_api"
```

4. **Uruchom aplikację:**
```bash
streamlit run app.py
```

## 📋 Wymagania

- Python 3.8+
- OpenAI API key
- Mikrofon (dla funkcji nagrywania)
- Połączenie internetowe

## 🛠️ Zależności

Główne biblioteki:
- `streamlit` - interfejs webowy
- `openai` - integracja z OpenAI API
- `speech_recognition` - rozpoznawanie mowy
- `Pillow` - generowanie obrazów fiszek
- `tiktoken` - liczenie tokenów

## 🎯 Jak używać

### Tłumaczenie
1. Wybierz język docelowy
2. Wpisz tekst lub nagraj audio
3. Kliknij "Przetłumacz"
4. Otrzymaj tłumaczenie z opcjonalnymi poprawkami

### Fiszki
1. Wpisz tekst do analizy
2. Wybierz język definicji
3. Kliknij "Wygeneruj fiszki"
4. Pobierz obraz fiszek jako PNG

### Ćwiczenie wymowy
1. Wybierz język i typ ćwiczeń
2. Kliknij "Generuj słowa do ćwiczeń"
3. Nagraj swoją wymowę
4. Otrzymaj analizę wymowy przez AI

## 🔧 Konfiguracja

### Streamlit
Plik `.streamlit/config.toml` zawiera ustawienia aplikacji:
- Port serwera
- Ustawienia cache
- Konfiguracja UI

### OpenAI
- Model domyślny: `gpt-4o`
- Maksymalne tokeny: 1200
- Temperatura: 0.7

## 📱 Interfejs

Aplikacja ma **polski interfejs** i jest zoptymalizowana pod kątem:
- **Responsywności** - działa na różnych urządzeniach
- **Intuicyjności** - prosty w użyciu
- **Wydajności** - szybkie odpowiedzi AI
- **Estetyki** - nowoczesny design

## 🎨 Funkcje wizualne

- **Gradientowe tła** i **cienie** dla fiszek
- **Kolorowe akcenty** i **ikony**
- **Responsywny layout** z kolumnami
- **Animowane spinnery** podczas ładowania

## 📊 Statystyki użycia

Aplikacja śledzi:
- **Liczbę tokenów** użytych
- **Koszty API** OpenAI
- **Historię żądań** z timestampami
- **Wydajność** różnych modeli

## 🤝 Wsparcie

W przypadku problemów:
1. Sprawdź połączenie internetowe
2. Zweryfikuj API key OpenAI
3. Sprawdź logi aplikacji
4. Upewnij się, że mikrofon działa

## 📄 Licencja

Projekt edukacyjny - do użytku osobistego i naukowego.

---

**Stworzone z ❤️ w Python i Streamlit**
