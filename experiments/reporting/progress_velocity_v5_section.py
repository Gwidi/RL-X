"""Prepared training with external-tracking-compatible velocity observations."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE='../artifacts/velocity_training_v5_preparation_2026-10-27/'

def build(table):
    d=json.loads((ROOT/'docs/artifacts/velocity_training_v5_preparation_2026-10-27/validation.json').read_text())
    return f'''<section id="e10" class="stage"><div class="stage-label">Etap 10 / 27.10.2026 / przygotowany trening v5</div>
<h2>Agent otrzymuje prędkość wzdłużną i boczną. Docelowe źródło na robocie: OptiTrack.</h2>
<p class="lead">Po potwierdzeniu dostępności prędkości z OptiTrack wdrożono dwa kanały aktora, profil v5 i konwersję zewnętrznego pomiaru do tej samej reprezentacji. test2.sh domyślnie uruchamia teraz ten trening od świeżych wag. Pełnego treningu na klastrze jeszcze nie uruchomiono; wykonano testy oraz krótki lokalny przebieg całego pipeline’u.</p>
<h3>Co dokładnie widzi polityka?</h3>
{table(['Element','Konfiguracja v5'],[
['Aktor','80 wejść: dotychczasowe 78 oraz v_forward i v_lateral'],
['Critic','Dotychczasowe 94 wejścia; nowe dwa kanały są wyłącznie w masce aktora'],
['Rama','Związana z kierunkiem bazy, wypoziomowana względem grawitacji: przód i lewy bok'],
['Punkt','Początek bazy/trunk; nie przesunięty site IMU ani dowolny marker'],
['Skala','Prędkość w m/s podzielona przez 4; bez clippingu dużych wartości'],
['Dostępność','Od pierwszego resetu, co krok sterowania 20 ms'],
['Źródło w symulacji','Rzeczywista prędkość bazy; model idealnego zewnętrznego pomiaru'],
['Źródło na robocie','Planowany OptiTrack, po kalibracji osi, punktu, orientacji i synchronizacji']])}
<p>Nie odsłonięto aktorowi dodatkowo wysokości ani prędkości pionowej. Dwa kanały są wyliczane w poziomej ramie, więc pionowy spadek przy pitch=10° nie staje się pozorną prędkością wzdłużną. Pole goal_velocities nadal oznacza komendę i nie zastępuje rzeczywistego pomiaru.</p>
<h3>Konwersja OptiTrack do wejścia sieci</h3>
<p>Funkcja tracking_velocity_features przyjmuje skalibrowaną prędkość śledzonego punktu w m/s, transformację jego układu do świata Z-up, orientację bazy oraz omega i przesunięcie znacznika. Odejmowana jest składowa omega×r, aby uzyskać prędkość początku bazy; następnie następuje projekcja na przód/bok i skalowanie /4. Jeżeli rigid-body-origin odpowiada już bazie, offset jest zerowy.</p>
<p>Jest to wspólny kontrakt i funkcja przeliczenia, nie klient NatNet ani wykonane połączenie z fizycznym systemem. Przed wdrożeniem należy uzgodnić timestampy, osie, orientację rigid body względem bazy oraz zmierzyć błąd i opóźnienie. Dostępność OptiTrack rozwiązuje wybór źródła informacji; nie wymaga zaczynania od estymatora opartego wyłącznie na IMU.</p>
<h3>Profil i kontrola A/B</h3>
{table(['Parametr','Wartość'],[
['PD / fizyka','PD606/home; implicitfast, 2,5 ms; sterowanie 50 Hz'],
['Zadanie','h=1–2,2 m; roll/pitch do ±10°; vx/vy do ±3 m/s na składową'],
['PPO / reward','Jak v4: std startowe 0,15, LR 3e-5, 10 epok; ta sama nagroda i limity'],
['Curriculum','Grupy 20/20/20/40, osobna trudność kątów i prędkości, wysokość niezależna'],
['Budżet','30 015 488 przejść, 64×2048; Slurm 12 h'],
['Domyślny tryb','heading_xy: oba kanały prędkości aktywne'],
['Kontrola','POLICY_VELOCITY_MODE=masked: dwa kanały zero, ale identyczne 80 wejść'],
['Checkpoint','Nowe wagi; kontrakt wejścia zapisany i sprawdzany przy wczytywaniu']])}
<p>Domyślny szum, bias i opóźnienie wynoszą zero. Są konfigurowalne niezależnie od randomizacji fizyki, ale nie nazywamy przykładowych nastaw kalibracją OptiTrack. Opóźnienie ma jednostkę kroków sterowania: 1=20 ms. Reset czyści historię; nie korzysta z przyszłych próbek. Osobny RNG sensora nie zmienia losowań fizyki. Maskowanie dotyczy aktora — critic i diagnostyka mogą nadal znać prędkość.</p>
<h3>Uruchomienie na klastrze</h3>
<pre><code>sbatch experiments/test2.sh</code></pre>
<p>W katalogu experiments: sbatch test2.sh. Wariant kontrolny:</p>
<pre><code>POLICY_VELOCITY_MODE=masked sbatch --export=ALL experiments/test2.sh</code></pre>
<p>Log: out_and_err_landing_seed606_velocity_v5_JOBID.txt. Modele: runs/soft_landing/landing_seed606_velocity_v5/seed606_velocity_v5_heading_xy_JOBID/models/. Wariant masked ma własną nazwę runu. Stare wagi v4 oraz wagi niezgodnego trybu, skali lub ustawień sensora są odrzucane, zamiast niejawnie zmieniać znaczenie wejść.</p>
<h3>Co sprawdzono?</h3>
<p>Finalna regresja: {d['tests']} testów, w tym 9 nowych dotyczących prędkości. Obejmują znaki i yaw, pionowy lot przy przechyle, kalibrację i lever arm, wymiary 80/94, brak clippingu, brak przecieku w masked, identyczną fizykę przy identycznych akcjach, noise/delay/reset/cache oraz zgodność checkpointów.</p>
<p>Rzeczywisty trening przez test2.sh wykonał {d['training_steps']} przejścia i {d['training_completed_episodes']} finalizacji epizodów, następnie końcową ewaluację {d['evaluation_episodes']} epizodów: cztery wysokości × cztery rodzaje warunków × dwa kontrolery × dwie próby. Sprawdzono zapis, odczyt i inferencję checkpointu oraz kontrakt we wszystkich wierszach CSV. To potwierdza działanie pipeline’u, a nie skuteczność wytrenowanej polityki v5.</p>
<p>Nowe logi pokazują rzeczywiste i obserwowane prędkości wzdłużne/boczne, ich moduł i błąd pomiaru. W wariancie idealnym obserwowane wartości są równe rzeczywistym. Do oceny naukowej pozostają pełny trening, wspólne zamrożone testy i powtórzenia po seedach; wynik v4 41,8% nie jest wynikiem nowego v5.</p>
<p><a href="../../experiments/landing_seed606_v5_README.md">Pełna instrukcja, OptiTrack i opcje sensora</a> · <a href="{BASE}profile.json">Profil v5</a> · <a href="{BASE}masked_profile.json">Profil kontroli masked</a> · <a href="{BASE}validation.json">Walidacja pipeline’u</a> · <a href="{BASE}tests.log">41 testów</a> · <a href="{BASE}paired_smoke.log">Log krótkiego treningu i ewaluacji</a></p></section>'''
