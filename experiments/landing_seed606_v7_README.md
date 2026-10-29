# V7 — v6 z szumem i opóźnieniem obserwacji prędkości

Uruchomienie z katalogu projektu:

```bash
sbatch experiments/test2.sh
```

Z `experiments`: `sbatch test2.sh`. Domyślny entrypoint:
`train_landing_seed606_v7.py`. Pełnego treningu nie uruchomiono lokalnie.

## Zakres zadania pozostaje jak w v6

Wysokość bazy 1–3 m (25% resetów dokładnie na 3 m), niezależne roll/pitch
stopniowane 1→30° na oś, moduł prędkości poziomej 0,5→5 m/s,
jednostajny rozkład po polu koła we wszystkich kierunkach.
Grupy curriculum nominalne/kąty/prędkość/razem: 20/20/20/40%.
Ewaluacja na 3 m: cztery warunki, polityka i zero akcji, po 64 epizody
na warunek/kontroler (512 epizodów na zestaw). Fizyka ewaluacji zamrożona.

PPO, PD606/home, reward, kontakt, integrator implicitfast 2,5 ms,
sterowanie 50 Hz i kryterium best są jak w v6. Budżet 30 015 488 przejść,
64×2048, Slurm 1 GPU / 16 CPU / 128 GB / 12 h.
Wynik v6 nie jest warunkiem uruchomienia v7.

## Jedyna zmiana merytoryczna: model pomiaru dwóch prędkości aktora

```text
noise_std_mps = 0.1   # σ na każdą składową świata XY, przed skalowaniem
bias_std_mps  = 0.0   # nie dodajemy stałego błędu epizodu
delay_steps   = 1    # 1 krok sterowania = 20 ms; nie krok fizyki 2,5 ms
```

Przy każdym nowym kroku sterowania dodawany jest niezależny, gaussowski
szum o średniej 0 i odchyleniu standardowym 0,1 m/s do każdej poziomej składowej prędkości bazy w świecie.
Próbka przechodzi przez bufor jednego kroku, następnie jest rzutowana na
aktualną poziomą ramę kierunku bazy (przód/lewy bok) i dzielona przez 4 m/s.
Szum cechy sieci przy braku zmiany kierunku ma więc σ=0,025.
Prędkość pionowa nie jest ujawniana aktorowi.

Pierwsza próbka przy resecie wypełnia bufor: nie zakładamy dostępności
historii sprzed epizodu i nie wstawiamy fikcyjnego pomiaru zero. Przez
pierwszy okres pomiar jest podtrzymywany. Potem obserwacja ma stałe
opóźnienie 20 ms; brak losowego jitteru, dropoutów i dodatkowego filtra.
Powtórne odczytanie obserwacji w tej samej chwili nie losuje ponownie szumu.

Osobny RNG sensora nie zmienia losowań fizyki. Szum i opóźnienie działają
od początku treningu i podczas ewaluacji, również przy zerowym współczynniku
randomizacji fizyki. Zmieniane są tylko dwa kanały aktora; critic ma nadal
94 dotychczasowe uprzywilejowane wejścia, a actor 80.

Jest to **założony scenariusz odporności**, nie zmierzona charakterystyka
OptiTrack. Przed robotem należy skalibrować układ współrzędnych, punkt
pomiaru, znaczniki czasu oraz zmierzyć rzeczywisty szum i opóźnienie.
Zapisany błąd obserwacji zawiera także efekt opóźnienia; w chwili uderzenia
może znacznie przekroczyć 0,1 m/s i nie świadczy to samo w sobie o błędzie RNG.

## Start i logi

V7 zaczyna **od nowych wag**. Nie wczytuje automatycznie modeli v6.
Kontrakt checkpointu obejmuje szum i opóźnienie, więc próba niejawnego
wczytania v6 do v7 jest odrzucana. Nie osłabiano tego zabezpieczenia.

Log: `out_and_err_landing_seed606_noisy_velocity_v7_JOBID.txt`.
Modele:
`runs/soft_landing/landing_seed606_noisy_velocity_v7/seed606_noisy_velocity_v7_heading_xy_JOBID/models/`.
Są oddzielone od v6. Latest zapisuje się niezależnie od kryterium best.

```bash
# Kontrola konfiguracji:
PYTHON_BIN=/sciezka/do/python bash experiments/test2.sh --check-only

# Kolejny seed:
sbatch experiments/test2.sh --environment.seed=2

# Kontrola masked (te same dwa miejsca wejścia, wartości zerowane):
POLICY_VELOCITY_MODE=masked sbatch --export=ALL experiments/test2.sh

# Historyczny v6; nazwy logów Slurm pozostają z launchera v7:
LANDING_TRAIN_ENTRY=train_landing_seed606_v6.py sbatch experiments/test2.sh
```

Telemetria `observation/velocity_*`: rzeczywiste i obserwowane składowe,
moduły oraz norma błędu. Polityka nie dostaje prawdziwych wartości przez te
logi. Metadane CSV zapisują pełny kontrakt sensora i `speed_disk`.

## Walidacja i wymagane pliki

Regresja: 34 testy, w tym 4 nowe sprawdzające niezmienność zadania,
identyczną fizykę przy tych samych akcjach, szum aktywny w ewaluacji,
bufor/cache/reset i odrzucanie checkpointów z innym modelem sensora.
Smoke: 1024 przejścia oraz 16 epizodów końcowej ewaluacji na 3 m.
Artefakty: `docs/artifacts/v7_preparation_2026-10-29/`.

Do stagingu dodano nowy profil, ten dokument, test v7 i `test2.sh`.
Zależności v5/v6, `policy_velocity.py` i sampler koła są już śledzone.
Nie wymagają raportowych skryptów, lokalnych modeli ani pobranych logów.
