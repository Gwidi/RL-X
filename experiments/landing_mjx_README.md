# Lądowanie na MJX-Warp (GPU)

To samo zadanie co CPU V22 (16 Nm, kręgosłup ±0,785 rad, 0,5–1,25 m, 0–5 m/s), symulowane na GPU.
Kod: `rl_x/environments/custom_mujoco/robot_locomotion/mjx_landing/` i `rl_x/algorithms/ppo/flax_mjx_landing/`.
Model fizyki, układ obserwacji i wartości nominalne pochodzą z CPU `LocomotionEnv`, więc definicja zadania jest jedna.

## Środowisko na klastrze

Nakładka venv na istniejący conda, bez zmieniania go (może z niego trenować inny job):

```bash
python -m venv --system-site-packages "$HOME/venvs/rlx_mjx"      # przy aktywnym conda rlx_filip
source "$HOME/venvs/rlx_mjx/bin/activate"
pip install mujoco==3.14.0 mujoco-mjx==3.14.0 dm_control==1.0.47 warp-lang==1.18.0 trimesh
```

`mujoco` i `mujoco-mjx` muszą mieć tę samą wersję. `dm_control` musi pasować do `mujoco`.
Sprawdzone lokalnie: powyższe wersje, jax 0.7.2, RTX 3050. Python 3.11 i H100 nie były sprawdzone.

## Start

`PYTHON_BIN` jest konieczny: bez niego skrypt aktywuje stare środowisko conda.

```bash
export PYTHON_BIN="$(which python)"
python experiments/train_landing_real_drives_mjx_v22.py --check-only     # na węźle z GPU
sbatch experiments/real_drives_mjx_v22.sh --runner.track_wandb=False     # od zera, 90 mln kroków
LANDING_V22_ARM=warm LANDING_V22_CHECKPOINT=/pelna/sciezka/v19/latest.model \
  sbatch experiments/real_drives_mjx_v22.sh --runner.track_wandb=False   # z V19, 60 mln kroków
```

**Kontynuacja V22 z V19** (`LANDING_V22_ARM=continue`): startuje z `latest.model` joba CPU 9730655 (60 030 976 kroków), dodaje
39 976 960 kroków na MJX, LR maleje liniowo z 3e-5 do 0. Skrypt sprawdza liczbę kroków, profil 16 Nm i kontrakt obserwacji.

```bash
LANDING_V22_ARM=continue LANDING_V22_CHECKPOINT=/pelna/sciezka/v22_warm/latest.model \
  sbatch experiments/real_drives_mjx_v22.sh --runner.track_wandb=False
```

LR jest częścią stanu Adama, więc MJX ustawia go przed każdą aktualizacją (`algorithm.final_learning_rate`; ujemne = stały).
Checkpoint zawsze zapisuje bazowe 3e-5, więc kolejny trening nie ruszy z zerowym LR.

Bez `--runner.track_wandb=False` potrzebny jest `WANDB_API_KEY` w terminalu, z którego idzie `sbatch`.

## Co jest takie samo

Nagroda, limity, kryteria sukcesu i porażki, rozkład startów, wyrzutnia i IMU (szum, opóźnienie 20 ms, historia 1 s),
akcje, sieci, hiperparametry PPO, batch 131 072, minibatch 64, 10 epok, siatka ewaluacji, reguła `best`.
Checkpointy wczytuje zwykłe `ppo.flax` i środowisko CPU.

## Co się różni

| | CPU V22 | MJX |
|---|---|---|
| Rollout | 64 środowiska × 2048 kroków | 1024 × 128 |
| Precyzja | float64 | float32 |
| Liczby losowe | NumPy | JAX |
| Reset | pełny na środowisko | 4096 gotowych stanów na iterację (`mjx.reset_pool_size`) |
| Rozgrzewka | brak | 256 kroków na środowisko bez uczenia |
| Ewaluacja | CPU, co 4,2 mln | GPU, te same komórki, inne losowania; zerowy sterownik liczony raz |
| Krytyk, 1. obserwacja | wysokość z końca poprzedniego epizodu | rzeczywista wysokość |

Randomizacja fizyki, mata, przejście do chodu i frontflip nie są zaimplementowane: konfiguracja z nimi kończy się błędem.

## Sprawdzenie zgodności

```bash
python experiments/analysis/check_mjx_parity.py --model modelsv19/latest.model       # krok po kroku
python experiments/analysis/compare_mjx_cpu_grid.py --model <checkpoint> --episodes 256   # siatka na obu silnikach
```

Drugi skrypt uruchom też na wytrenowanym modelu **przed wdrożeniem**: ewaluacja w treningu jest na tej samej fizyce co trening.

## Limit iteracji solvera

Flaga Warp `ITERATIONS`/`LS_ITERATIONS` oznacza, że solver nie zbiegł w limicie iteracji, a nie przepełnienie bufora.
Nie przerywa treningu; `rollout/solver_limit_worlds` liczy takie światy (narastająco). Przepełnienie bufora nadal przerywa.
