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

**Niedoskonałe wypuszczenie** (`LANDING_V22_ARM=disturbed`): roll i pitch ~ U(−30°, 30°) niezależnie, prędkość kątowa ciała
o losowym kierunku i module ~ U(0, 15°/s), do tego prędkość pozioma i wysokość jak w V22. Moduł zaburzeń rośnie liniowo od 0
przez pierwsze 20 054 016 kroków (`rollout/disturbance_scale`), ewaluacja zawsze z pełnym zaburzeniem we wszystkich komórkach,
także przy 0 m/s i dla zerowego sterownika. LR maleje z 3e-5 do 1e-5, 60 030 976 kroków. Rodzic: dowolny `latest.model` V22 (CPU lub MJX).
Przechył osi z robota może sięgać ok. 42°, bo roll i pitch są losowane osobno. Nie ma randomizacji fizyki.
Bramka `nominal_eligible` wymaga >80% przy 0 m/s, już z zaburzeniem; dopóki jej nie spełnia, `best.model` się nie zapisze (`latest` zapisuje się zawsze).

```bash
LANDING_V22_ARM=disturbed LANDING_V22_CHECKPOINT=/pelna/sciezka/latest.model \
  sbatch experiments/real_drives_mjx_v22.sh --runner.track_wandb=False
```

Walidacja na CPU z tym samym zaburzeniem: `compare_mjx_cpu_grid.py --model ... --tilt 30 --rate 15`.

**Start z pozy stania** (`LANDING_V22_ARM=standing`): wypuszczenie z symetrycznej pozy stania (kręgosłup 0, hipy ∓0,1, uda ±0,8,
łydki ±1,5; `STAND_POSE` w `environment.py`) zamiast z pozy lądowania. Poza wchodzi stopniowo: interpolacja od pozy lądowania do stania
przez pierwsze 30 015 488 kroków (`rollout/pose_scale`), 20% epizodów zostaje na pozie lądowania. Ewaluacja zawsze ze stania we wszystkich
komórkach. 90 046 464 kroków, LR 3e-5 maleje do 1e-5. Rodzic: dowolny `latest.model` V22; sensownie ostatni z kontynuacji V19.
Reguły sukcesu i kontaktu bez zmian. Nie ma jeszcze walidacji na CPU dla tego ramienia: środowisko CPU nie startuje ze stania.
Niewytrenowany rodzic ma ze stania ok. 0–6% sukcesu (`docs/artifacts/v22_contact_diagnostic_2026-11-25`).

```bash
LANDING_V22_ARM=standing LANDING_V22_CHECKPOINT=/pelna/sciezka/latest.model \
  sbatch experiments/real_drives_mjx_v22.sh --runner.track_wandb=False
```

**Miękki kontakt uda i łydki** (`LANDING_V22_ARM=soft`): dotyk uda i łydki (ogniwa l1, l2) jest dozwolony, dopóki suma siły normalnej
tych kontaktów uśredniona po 10 ms nie przekroczy 200 N (`mjx.limb_contact.force_limit`, ok. 1,5 × ciężar robota). Powyżej próba kończy się
jak przy każdym niedozwolonym kontakcie (w logach jako `disallowed_contact`). Biodro, tułów i tylny korpus nadal kończą próbę.
`env_info/metrics/peak_limb_force_10ms` i `eval/grid/.../peak_limb_force_10ms` pokazują siłę. Pomiar, z którego wzięto próg, jest w
`docs/artifacts/v22_contact_diagnostic_2026-11-25`: przy 100–200 N odzyskuje się ok. 90% prób, powyżej 400 N prawie żadna.
`LANDING_V22_REFERENCE` wybiera człon hamowania w nagrodzie: `track` (domyślnie, jak dotąd), `upper` (kara tylko za prędkość powyżej referencji,
hamować szybciej wolno) albo `off` (brak członu). 60 030 976 kroków, LR 3e-5 maleje do 1e-5, rodzic: dowolny `latest.model` V22.
Niewytrenowany rodzic nie zyskuje na tej regule (przy 5 m/s 39,5%, tyle samo co ze ścisłą), bo nie uczył się lekkich dotyków; zysk ma dać dopiero trening.
Bez walidacji na CPU: środowisko CPU nie ma tej reguły.

```bash
LANDING_V22_ARM=soft LANDING_V22_REFERENCE=track LANDING_V22_CHECKPOINT=/pelna/sciezka/latest.model \
  sbatch experiments/real_drives_mjx_v22.sh --runner.track_wandb=False
```

**Kara za poślizg** (`LANDING_V22_SLIP_SCALE`, dowolne ramię): mnożnik kary za poślizg stóp pod obciążeniem. `0` ją usuwa, `1` zostawia
(domyślnie). Waga w konfiguracji CPU zostaje 5, więc checkpointy pozostają zgodne; mnożnik jest tylko w `mjx.slip_penalty_scale`.
`loaded_slip_integral` jest nadal mierzony i logowany, więc widać, czy polityka nauczyła się ślizgać. Uwaga: tarcie w symulacji jest stałe i
nie jest randomizowane, więc nauczone ślizganie może nie przenieść się na inną podłogę.

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
