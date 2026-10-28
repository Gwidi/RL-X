# V6 — prędkość pozioma do 5 m/s, roll/pitch do 30°, wysokość do 3 m

Uruchomienie z katalogu projektu:

```bash
sbatch experiments/test2.sh
```

Z `experiments`: `sbatch test2.sh`. Domyślny entrypoint to teraz
`train_landing_seed606_v6.py`. Zachowano zasoby Slurm: 1 GPU, 16 CPU,
128 GB, limit 12 h. Pełnego zadania nie wysłano z lokalnego komputera.

## Dokładne znaczenie nowych zakresów

- Wysokość początku bazy nad terenem: **1–3 m**, niezależnie od trudności
  kątów i prędkości. 25% resetów dokładnie na 3 m, pozostałe jednostajnie
  w przedziale 1–3 m. Nie dodajemy wysokości nominalnej robota.
- Roll i pitch niezależnie od zera: docelowo każdy w **[−30°, +30°]**.
  Przy jednoczesnych skrajnych roll/pitch całkowity przechył może przekroczyć
  30° — ograniczenie dotyczy każdej osi, nie pojedynczego kąta od pionu.
- **Moduł prędkości poziomej ≤5 m/s**, we wszystkich kierunkach.
  Nie losujemy osobno vx/vy do ±5, co dawałoby moduł 7,07 m/s.
  Nowy tryb `velocity_sampling=speed_disk` losuje jednostajnie po polu koła:
  `speed=limit*sqrt(U)`, `azimuth=uniform(0,2π)`.
  Jest to rozkład o większym udziale dużych prędkości niż jednostajny moduł.
- Początkowe vz=0 i prędkości kątowe=0. Yaw=0.
- Grupy: nominalnie 20%, same kąty 20%, sama prędkość 20%, oba czynniki 40%.
- Niezależne curriculum: kąty **1→30°**, moduł prędkości **0,5→5 m/s**.
  Wysokość pozostaje niezależna; 3 m jest dostępne od początku.

## Ewaluacja i wybór modelu

Stała ewaluacja wyłącznie na **3 m** (baza nad terenem):
nominalnie, same kąty ±30°, sama prędkość w kole do 5 m/s, oba czynniki.
W każdej grupie porównywane są polityka i zero akcji przy identycznym
scenariuszu. Trening i ewaluacja używają tej samej funkcji losowania.
W CSV zapisany jest tryb `velocity_sampling`.

Po 64 epizody na warunek i kontroler: 512 epizodów na pełny zestaw.
Nominalne próby bez losowania są deterministycznymi powtórzeniami,
nie 64 niezależnymi konfiguracjami. Ewaluacja co 8 podstawowych interwałów
PPO i po zakończeniu treningu, jak w v5.

Dotychczasowy warunek dla best pozostaje: 100% nominalnego lądowania,
a następnie ocena skuteczności przy połączonych zaburzeniach.
Przy jednej wysokości metryka worst_height oznacza wynik na 3 m.
Jeśli warunek nominalny nie zostanie spełniony, best może nie powstać;
latest jest zapisywany niezależnie.

## Co zachowano i jak startuje trening

V6 domyślnie zaczyna **od nowych wag**, tak jak v5. Nie wczytuje automatycznie
root `latest.model` ani pliku zależnego od lokalnych pobrań.
Nie jest to wznowienie liczników ani optymalizatora zakończonego v5.

Aktor 80 wejść (w tym rzeczywista prędkość przód/bok), critic 94, skala
prędkości **/4 m/s** bez clippingu. Przy 5 m/s pojedyncze wejście może
osiągnąć ±1,25. W symulacji nadal idealny pomiar, bez opóźnienia i szumu.

Bez zmian: PD606/home, 13 akcji, skala 0,6 rad, reward i kryteria sukcesu,
kontakt dolnej 1/3 i zakaz górnej 2/3 łydki, implicitfast 2,5 ms,
sterowanie 50 Hz, std początkowe 0,15, LR 3e-5, 10 epok PPO.
Budżet: 30 015 488 przejść, 64 środowiska × 2048 kroków na rollout.
To pierwszy budżet dla trudniejszej domeny, nie gwarancja ukończenia curriculum.

Historyczny `component_box` pozostaje domyślny poza v6 i zachowuje kolejność
losowań v4/v5. Zmienia się domena i rozkład, dlatego v5/v6 nie są izolowaną
ablacją jednego hiperparametru.

## Warianty i pliki

```bash
# Kontrola obserwacji prędkości przy tej samej architekturze:
POLICY_VELOCITY_MODE=masked sbatch --export=ALL experiments/test2.sh

# Inny seed:
sbatch experiments/test2.sh --environment.seed=2

# Jawny powrót do v5 (nazwy nagłówków Slurm trzeba nadpisać, jeśli mają być v5):
LANDING_TRAIN_ENTRY=train_landing_seed606_v5.py sbatch experiments/test2.sh

# Tylko profil, bez treningu:
PYTHON_BIN=/sciezka/do/python bash experiments/test2.sh --check-only
```

Log: `out_and_err_landing_seed606_extended_v6_JOBID.txt`.
Modele:
`runs/soft_landing/landing_seed606_extended_v6/seed606_extended_v6_heading_xy_JOBID/models/`.
Wariant masked ma osobny run. Przenosząc kod na klaster należy uwzględnić
nowy profil, wspólny sampler `horizontal_velocity.py`, zmiany początkowego
stanu, domyślnej konfiguracji, wrappera ewaluacji i `test2.sh`.

## Preflight i walidacja

Przed treningiem odtworzono 26 prób diagnostycznych: PD/zero i v5 latest,
trzy wysokości nominalne oraz przykłady skrajnych kątów/prędkości na 3 m.
PD zaliczył nominalne 3 m, a v5 latest zakończył tę próbę niedozwolonym
kontaktem. Przypadki graniczne nie stanowią reprezentatywnego estymatora
skuteczności i nie dowodzą, że v6 opanuje nową domenę.

Wyniki testów i lokalnego krótkiego treningu:
`docs/artifacts/v6_preparation_2026-10-28/validation.json`.
