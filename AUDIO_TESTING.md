# Test własnych nagrań

W `testdata/real_audio_tests/` są cztery pary AAC. Plik `_r` jest nagraniem referencyjnym, a plik bez `_r` jest weryfikowany.

| Para | Fraza | Oczekiwany wynik |
|---|---|---|
| p1 | chmura rower książka wiatr | akceptacja: ten sam mówca i fraza |
| p2 | fortuna często rozumowi przeczy | odrzucenie: fraza nie pasuje |
| p3 | gdzie drwa rąbią, tam wióry lecą | akceptacja: ten sam mówca i fraza |
| p4 | w nocy wszystkie koty są szare | odrzucenie: inny mówca |

Uruchom test z katalogu projektu:

```powershell
docker compose up -d --build
python .\scripts\run_real_audio_tests.py
```

Domyślny adres API to `http://localhost:8010`, a token to `dev-token`. Skrypt sprawdza osobno oczekiwaną decyzję, zgodność mówcy i frazy. Szczegóły zapisuje do `testdata/real_audio_tests/results.json`.

## PAD i interpretacja wyników

AASIST analizuje każdą próbkę, ale jego wynik jest domyślnie diagnostyczny (`PAD_ENFORCED=false`) i nie blokuje akceptacji. W tych testach AASIST nadał bardzo niskie wyniki także dwóm prawidłowym, czystym nagraniom (p1: 0,003; p3: 0,459 przy progu 0,90). Wymuszenie obecnego progu odrzucałoby prawidłowe próby, więc PAD nie jest jeszcze skalibrowany dla tego mikrofonu/kodeka/domeny.

Żaden z czterech przykładów nie jest potwierdzoną próbą TTS/deepfake ani odtworzeniem nagrania. p4 to inny żywy mówca i jest odrzucany przez ECAPA. Te nagrania sprawdzają VAD, frazę i tożsamość mówcy, ale nie dowodzą skuteczności wykrywania spoofingu. Przed włączeniem PAD zbierz oznaczone próbki prawdziwe, TTS/voice-cloning i odtworzenia, a próg dobierz na osobnym zbiorze walidacyjnym.

Po kalibracji można jawnie włączyć twardą bramkę w PowerShell:

```powershell
$env:PAD_ENFORCED = "true"
docker compose up -d
curl.exe http://localhost:8010/health
```

Odpowiedź health pokaże wtedy `pad_enforced: true`. Dla testu czterech par ustaw `PAD_ENFORCED=false` (lub otwórz nową sesję PowerShell), a następnie ponownie wykonaj `docker compose up -d`.

