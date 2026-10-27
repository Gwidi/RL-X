# V4 — trening po korekcie numerycznej PD

Uruchomienie na klastrze z katalogu projektu:

```bash
sbatch experiments/test2.sh
```

Z katalogu `experiments`: `sbatch test2.sh`. Domyślnym wejściem jest teraz
`train_landing_seed606_v4.py`. Wspólna aktywacja Condy nadal wykonuje hooki
bez `nounset`; EGL jest ustawiane przez launcher.

Kontrola lokalna przed wysłaniem:

```bash
PYTHON_BIN=/pełna/ścieżka/do/python bash experiments/test2.sh --check-only
```

## Dlaczego ta konfiguracja

W replayach tych samych komend mniejsze Kd usunęło oscylacje momentu łydki,
ale obniżanie gainów nie dało wyraźnej poprawy całych lądowań obecnych wag.
Integrator `implicitfast` usuwa naprzemienne nasycenie bez redukcji oryginalnego
PD. W v4 przyjmujemy `implicitfast`, fizykę 2.5 ms i sterowanie nadal 20 ms
(8 podkroków). Sprawdzono także 5 i 1.25 ms; wyniki detekcji kontaktu i
sukcesu nie są w pełni niezależne od kroku — to nie jest deklaracja ścisłej
zbieżności fizycznej kontaktu.

## Zakres pierwszego treningu v4

Rewizja 2026-10-27: trening od świeżych wag po korekcie fizyki, z maksymalną
wysokością zmniejszoną do 2.2 m i celem przechyłu zwiększonym do ±10°.
Względem v3 zmienia się także domena zadania, więc porównanie zagregowanych
wyników nie izoluje już samej zmiany integratora.
Nastawy PD606, home, skala akcji 0.6 rad, 13 akcji, limity, contact/reward,
obserwacje i PPO (std=0.15, LR=3e-5, 10 epok) pozostają takie jak w v3.
Wysokość 1–2.2 m: 25% resetów dokładnie 2.2 m, 75% U[1;2.2] m.
Grupy 20/20/20/40, roll i pitch od początkowego zakresu ±1° do docelowego
±10° na każdą oś; vx/vy docelowo
±3 m/s na składową. Nie dodano szumu skorelowanego ani wyboru Kd przez aktora.
Ewaluacja: wysokości 1.0/1.5/2.0/2.2 m i pełne ±10° na każdą oś.
30–45° pozostaje poza tym treningiem.

Budżet 30 015 488 przejść, 64 env × 2048. Limit Slurm zwiększono z 7:30 do 12 h;
przy drobniejszym kroku symulacja wymaga większej liczby podkroków.
Można jawnie dobrać czas do limitów partycji przez `sbatch --time=...`.
Nie wznawiać v3 na nowej fizyce jako tego samego eksperymentu.
Zapis profilu w checkpointach i kontrola jego SHA blokują niezgodne resume.

Weryfikacja około 1 mln przejść: nominalne wyniki, powrót oscylacji,
metryki curriculum i KL/clip fraction. Około 10 mln: porównać z v3 na tej
samej domenie, ocenić mapy wysokości i zaburzeń. Wyniki rolloutów nie
zastępują pełnej, stałej walidacji.

Nowe pasywne metryki (bez wpływu na reward):
- `env_info/metrics/flight_calf_saturation_fraction`;
- `env_info/metrics/flight_calf_sign_flip_fraction` (odwrócenia powyżej 10% limitu);
- `env_info/metrics/flight_motor_saturation_fraction`;
- `env_info/metrics/flight_motor_samples`.

Próbkowanie przed pierwszym wykrytym kontaktem z podporą, co 2.5 ms.
Metryki konsoli są agregatami rolloutów; do dokładnej diagnostyki epizodu
używać zapisanych przebiegów, a nie interpretować agregat jako procent epizodów.
Nasycenie podczas szybkiego celowego ruchu może być fizycznie poprawne;
sam niezerowy wskaźnik nasycenia nie dowodzi powrotu oscylacji.

Modele:
`runs/soft_landing/landing_seed606_stable_v4/seed606_stable_v4_JOBID/models/`.
Log: `out_and_err_landing_seed606_stable_v4_JOBID.txt`.
`best` nadal wymaga 100% nominalnie i jest wybierany wg najgorszej wysokości
połączonych zaburzeń; `latest` zachowuje bieżące wagi. Walidacja zachowuje rodzaje warunków v3, lecz używa nowej fizyki,
wysokości do 2.2 m i kątów do ±10°. Końcowy test artykułu wymaga
niezależnych scenariuszy oraz kolejnych seedów treningu.

## Zgodność z search

Wspólny plik (skopiować na klaster razem z nowym kodem):
`rl_x/environments/custom_mujoco/robot_locomotion/robots/silver_badger/data/landing_physics_v4.json`.

Search CPU i GPU przyjmują:

```bash
python search/main.py --physics-profile rl_x/environments/custom_mujoco/robot_locomotion/robots/silver_badger/data/landing_physics_v4.json [pozostałe opcje]
python search/main_gpu.py --physics-profile rl_x/environments/custom_mujoco/robot_locomotion/robots/silver_badger/data/landing_physics_v4.json [pozostałe opcje]
```

Alternatywnie `RLX_LANDING_PHYSICS_PROFILE=/absolutna/sciezka/profil.json`;
zmienna jest dziedziczona przez procesy workerów CPU. Bez flagi pozostaje
stara fizyka — stare searche należy jawnie odtwarzać bez profilu v4.
Search nadal optymalizuje własne gainy/home; wspólny profil ustawia jedynie
integrator, krok i solver. Porównanie baseline wymaga jawnego podania PD606/home606.

Test CPU search–PPO sprawdza macierze modelu, 50 Hz, okna 10/50 ms i zgodność
trajektorii przez kontakt. Lokalnie brak pakietu mujoco-mjx: ścieżka GPU
otrzymuje wspólny profil, ale wykonania MJX nie zweryfikowano na tym komputerze.

Powrót do wcześniejszego launchera:
`LANDING_TRAIN_ENTRY=train_landing_seed606_factorized.py sbatch experiments/test2.sh`.
Historia i stare profile nie są nadpisywane.

## Odtworzenie diagnozy

```bash
rlx/bin/python experiments/check_pd_stability.py
rlx/bin/python experiments/evaluate_pd_variants.py --variants kd_075 calf_kd_half implicit_5ms implicit_2p5ms implicit_1p25ms
rlx/bin/python -m experiments.summarize_pd_stability
rlx/bin/python experiments/build_progress_report.py
```

Wyniki: `docs/artifacts/pd_stability_v4/`. Etap 7 raportu zawiera także
screening mocniejszych redukcji wzmocnień. Przebiegi są sparowane po stanie
początkowym. Transfer istniejących wag do nowej fizyki jest diagnostyką,
a nie wynikiem treningu v4.
