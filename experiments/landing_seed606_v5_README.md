# V5 — PPO z prędkością wzdłużną i boczną, docelowo z OptiTrack

## Uruchomienie na klastrze

Z katalogu projektu:

```bash
sbatch experiments/test2.sh
```

Z `experiments`: `sbatch test2.sh`. Domyślny entrypoint:
`train_landing_seed606_v5.py`, tryb `heading_xy`. Aktywacja Condy z poprawką
nounset oraz EGL pozostają w `train_landing_seed606.sh`.
Limit Slurm: 12 h; CPU 16, RAM 128 GB, jedna GPU.

Sprawdzenie konfiguracji przed wysłaniem:

```bash
PYTHON_BIN=/ścieżka/do/python bash experiments/test2.sh --check-only
```

Trening kontrolny z dwiema zerowanymi cechami, identycznym wymiarem sieci:

```bash
POLICY_VELOCITY_MODE=masked sbatch --export=ALL experiments/test2.sh
```

Dodatkowe seedy: `sbatch experiments/test2.sh --environment.seed=2` itd.
Pilot 4 194 304 przejść:
`sbatch experiments/test2.sh --algorithm.total_timesteps=4194304`.
Profile historyczne nadal można uruchomić jawnie:
`LANDING_TRAIN_ENTRY=train_landing_seed606_v4.py sbatch experiments/test2.sh`.

## Konkretna zmiana względem v4

- Aktor: 78 → 80 wejść; dodano `v_forward`, `v_lateral`.
- Critic: nadal 94 wejścia, dotychczasowa prędkość i pozostałe dane.
- Punkt odniesienia: początek bazy/trunk, a nie przesunięty site IMU.
- Układ: kierunek robota wypoziomowany względem grawitacji; +x do przodu,
  +y w lewo, Z świata ku górze. Przy yaw=0 odpowiada XY świata.
- Cechy sieci = prędkości w m/s / 4. Nie ma clippingu prędkości do ±1.
- Sygnał dostępny od pierwszej obserwacji, następnie co 20 ms.
- W symulacji używana jest dokładna prędkość bazy. To założenie idealnego
  zewnętrznego pomiaru, zgodnego z planowanym użyciem OptiTrack; nie wynik
  estymatora uczonego ani pomiar zwykłej IMU.

Nastawy PD606, home, kontakt, reward, 13 akcji, skala 0.6 rad, fizyka
implicitfast 2.5 ms, sterowanie 50 Hz, PPO/std=0.15/LR=3e-5/10 epok,
wysokości 1–2.2 m, kąty do ±10° i vx/vy do ±3 m/s są jak w v4.
Grupy curriculum 20/20/20/40 i zasada wyboru najlepszego checkpointu
pozostają takie same. Domyślny budżet: 30 015 488 przejść, 64×2048.

Wagi są świeże. Stare best/latest z v4 mają inny kontrakt wejścia i są
odrzucane przy próbie wczytania. `masked` i `heading_xy` mają oba 80 wejść,
ale również nie wolno ich niejawnie zamieniać przy wczytywaniu wag.
Kontrakt zapisany w config_environment.json obejmuje wersję, tryb, skalę,
punkt i ramę odniesienia oraz ustawienia modelu szumu/opóźnienia.

## OptiTrack — kontrakt do wdrożenia

Przygotowano czystą funkcję `tracking_velocity_features` w
`rl_x/environments/custom_mujoco/robot_locomotion/mujoco/policy_velocity.py`.
Nie jest to klient NatNet ani połączenie z rzeczywistym systemem; adapter
sterownika musi dostarczyć poprawnie zsynchronizowane dane.

```python
features = tracking_velocity_features(
    velocity_tracking=v_tracking_mps,
    world_from_tracking=R_WT,
    world_from_base=R_WB,
    angular_velocity_world=omega_world,
    marker_offset_base=r_base_to_marker,
    scale_mps=4.0,
)
```

`R_WT` jest skalibrowaną transformacją wektorów z osi systemu śledzenia
do prawoskrętnego świata Z-up. Nie zakładać automatycznie macierzy jednostkowej.
`R_WB` jest orientacją **bazy robota**, po uwzględnieniu kalibracji względem
rigid body OptiTrack. Prędkość ma być w m/s, omega w rad/s, offset w metrach.
Wszystkie dane muszą odnosić się do zgodnych chwil pomiaru.

Jeśli prędkość dotyczy markera/rigid-body-origin przesuniętego względem bazy,
stosujemy v_base = R_WT*v_marker − omega_world × (R_WB*r_base_to_marker).
Jeżeli OptiTrack już podaje prędkość początku bazy, offset wynosi zero.
Następnie wyliczane są dwie poziome cechy względem kierunku bazy i skala /4.
Nie podajemy aktorowi przy okazji prędkości pionowej ani wysokości.

Przed robotem: skalibrować osie, body-frame, offset i synchronizację; zmierzyć
błąd, opóźnienie i częstotliwość aktualizacji. Dopiero wtedy oceniać zgodność
rzeczywistych obserwacji z założeniem użytym w treningu. Nie testowano tu
połączenia z fizycznym OptiTrack ani wdrożenia na robocie.

## Szum i opóźnienie — jawne opcje

Domyślnie zero; nie wymyślamy parametrów konkretnej instalacji OptiTrack.
Możliwe nadpisania, np. po pomiarze toru:

```bash
sbatch experiments/test2.sh \
  --environment.policy_velocity.noise_std_mps=0.1 \
  --environment.policy_velocity.bias_std_mps=0.05 \
  --environment.policy_velocity.delay_steps=1
```

Są to tylko przykładowe ustawienia, nie kalibracja. Delay=1 oznacza 20 ms.
Szum jest niezależny od współczynnika randomizacji fizyki (działa także przy
nominalnej fizyce). Bias XY świata jest stały w epizodzie; biały szum jest
losowany na próbkę. Opóźniona prędkość świata jest przeliczana do aktualnej
ramy kierunku. Bufor po resecie trzyma pierwszy pomiar — nie ma dostępu do
próbek sprzed epizodu ani do przyszłości. Wielokrotny odczyt tej samej chwili
nie losuje szumu ponownie. RNG pomiaru nie przesuwa RNG dynamiki/resetów.
Zmiana tych opcji zmienia kontrakt checkpointu; wczytywanie do ewaluacji lub
resume wymaga zgodnej konfiguracji.

## Logi i modele

Log: `out_and_err_landing_seed606_velocity_v5_JOBID.txt`.
Wagi domyślnego wariantu:
`runs/soft_landing/landing_seed606_velocity_v5/seed606_velocity_v5_heading_xy_JOBID/models/`.
Kontrola: analogicznie `seed606_velocity_v5_masked_JOBID`.

Nowe metryki w m/s (przed skalą wejścia):
- `env_info/observation/velocity_forward_true`, `velocity_lateral_true`;
- `env_info/observation/velocity_forward_observed`, `velocity_lateral_observed`;
- `env_info/observation/velocity_speed_true`, `velocity_speed_observed`;
- `env_info/observation/velocity_error_norm`.

W trybie masked dwa kanały aktora są zerowe; rzeczywista prędkość może
pozostawać w diagnostyce i u krytyka. Nie jest to przeciek do wejścia aktora.
CSV ewaluacji zapisuje tryb i kontrakt; hash stanu dotyczy fizycznego resetu,
więc parowanie wariantów informacyjnych pozostaje możliwe.

Ocena po około 1 mln: walidacja, nominalne lądowania, telemetria prędkości,
brak wzorca naprzemiennego nasycenia; po 4–10 mln: trend w grupie velocity
oraz combined. Porównanie końcowe na wspólnych stanach i kilku seedach.
Nie interpretować krótkiego smoke jako skuteczności wytrenowanego v5.

## Testy przygotowania

```bash
rlx/bin/python -m unittest discover -s search -p 'test_*.py' -v
```

Testy sprawdzają osie i yaw, czysto pionowy ruch przy przechyle, kalibrację
i lever arm OptiTrack, brak clippingu, wymiary 80/94, brak przecieku do
masked, identyczną fizykę przy tych samych akcjach, reset/caching/szum/delay
oraz blokadę niezgodnych checkpointów. Krótki trening i ewaluacja są
archiwizowane w `docs/artifacts/velocity_training_v5_preparation_2026-10-27/`.
