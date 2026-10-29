"""Prepared extended-domain v6; validation is not a full training result."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v6_preparation_2026-10-28'
BASE='../artifacts/v6_preparation_2026-10-28/'
def build(table):
 d=json.loads((OUT/'validation.json').read_text())
 return f'''<section id="e12" class="stage"><div class="stage-label">Etap 12 / 28.10.2026 / przygotowany trening v6</div>
<h2>Nowa domena: do 5 m/s, ±30° na oś i 3 m wysokości.</h2>
<p class="lead">Przygotowano świeży trening v6. Użytkownik potwierdził limit modułu prędkości poziomej 5 m/s we wszystkich kierunkach. test2.sh uruchamia nowy profil; pełnego zadania nie wysłano na klaster. Wyników 92,6% i 96,1% v5 nie przenosimy na tę większą domenę.</p>
{table(['Element','V6'],[['Wysokość bazy','1–3 m niezależnie od trudności; 25% dokładnie na 3 m'],['Roll i pitch','Niezależne kąty ±30° na każdą oś; curriculum od ±1°'],['Prędkość','Koło vx²+vy² ≤25; kierunek dowolny; curriculum limitu od 0,5 do 5 m/s'],['Rozkład prędkości','Jednostajny po polu koła: r=L√U, azymut jednostajny 0–2π'],['Grupy','Nominalnie / kąty / prędkość / razem: 20 / 20 / 20 / 40%'],['Stała ewaluacja','Wyłącznie 3 m; nominalnie, kąty, prędkość, oba czynniki; pary polityka/zero'],['Obserwacja','Aktor 80, critic 94; pomiar przód/bok /4 m/s bez clippingu'],['Start i budżet','Nowe wagi, 30 015 488 przejść, 64×2048; Slurm 12 h']])}
<h3>Co dokładnie zmieniono względem v5?</h3>
<p>Oprócz większych zakresów zmieniono geometrię rozkładu prędkości: v5 losował niezależne vx i vy w kwadracie ±3 m/s, v6 losuje wektor w kole o promieniu 5 m/s. Dzięki temu 5 m/s jest rzeczywistym limitem modułu, a nie limitem składowej pozwalającym na 7,07 m/s po przekątnej. Rozkład jednostajny po polu koła częściej wybiera duże moduły niż jednostajny moduł. Identyczna funkcja obsługuje trening i ewaluację.</p>
<p>Historyczny tryb component_box pozostaje domyślny poza v6 i zachowuje dokładną kolejność losowań. Nowy tryb speed_disk jest zapisywany w konfiguracji i CSV ewaluacji. Maksymalne jednoczesne roll=pitch=30° oznaczają przechył od pionu większy niż 30°; nie jest to pojedynczy stożek orientacji o kącie 30°.</p>
<p>Wysokość 3 m pojawia się od początku treningu, niezależnie od kątów i prędkości. Curriculum stopniuje te dwa czynniki osobno. Początkowe vz i prędkości kątowe pozostają zerowe. Zasady aktualizacji trudności, PD606/home, 13 akcji, reward, kryteria bezpieczeństwa, implicitfast 2,5 ms i sterowanie 50 Hz są jak w v5. Skala obserwacji /4 nie ogranicza wartości: przy 5 m/s pojedyncza cecha może wynosić 1,25.</p>
<h3>Ewaluacja na 3 m</h3>
<p>Pełny zestaw: 64 epizody na warunek i kontroler, czyli 512 epizodów. Osobno same kąty, sama prędkość i połączone zaburzenia. Dobór best nadal wymaga 100% nominalnych sukcesów; przy jednej wysokości worst_height oznacza wynik na 3 m. Latest jest zapisywany niezależnie. Dotychczasowe wyniki na 1–2,2 m pozostają w etapie 11; nie porównujemy ich bezpośrednio jako izolowanej ablacji jednego parametru.</p>
<h3>Preflight: większa domena wymaga nowego manewru</h3>
<p>26 prób diagnostycznych PD/zero i v5 latest obejmuje trzy nominalne wysokości oraz wybrane skrajne kąty i prędkości. PD zaliczył nominalne 3 m, v5 latest przy tej wysokości zakończył próbę niedozwolonym kontaktem. Oznacza to, że wysokość 3 m nie jest z definicji niemożliwa w bieżącej fizyce, ale dotychczasowa polityka nie przenosi się automatycznie poza trening. Próby graniczne nie są estymatorem nowej skuteczności.</p>
<h3>Walidacja i uruchomienie</h3>
<p>Przeszło {d['tests']} testów, w tym 4 nowe: dokładna zgodność historycznych losowań, rzeczywiste resety na krańcach curriculum, zgodność par ewaluacyjnych i odrzucanie nieznanego rozkładu. Lokalny smoke wykonał {d['training_steps']} przejścia i końcowy zestaw {d['evaluation_episodes']} epizodów na 3 m. Potwierdzono zapis checkpointu i CSV. Jest to kontrola działania, nie wynik wytrenowanego v6.</p>
<pre><code>sbatch experiments/test2.sh</code></pre>
<p>Log: out_and_err_landing_seed606_extended_v6_JOBID.txt. Modele: runs/soft_landing/landing_seed606_extended_v6/seed606_extended_v6_heading_xy_JOBID/models/. Z katalogu experiments: sbatch test2.sh. Start jest od nowych wag; nie wczytuje automatycznie pobranego latest.model.</p>
<p><a href="../../experiments/landing_seed606_v6_README.md">Instrukcja v6 i warianty</a> · <a href="{BASE}profile.json">Pełny profil</a> · <a href="{BASE}validation.json">Walidacja</a> · <a href="{BASE}tests.log">30 testów</a> · <a href="{BASE}preflight.json">26 prób diagnostycznych</a> · <a href="{BASE}smoke.log">Log krótkiego treningu</a> · <a href="{BASE}landing_evaluation/1024_episodes.csv">Epizody końcowej ewaluacji smoke</a></p></section>'''

def overview():
 return '''<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan / przygotowany v6</div><h2>Rozszerzamy domenę do 5 m/s, ±30° i 3 m.</h2><p class="lead">V6 jest gotowy do świeżego treningu przez test2.sh. Limit dotyczy modułu prędkości poziomej we wszystkich kierunkach. Ewaluacja odbywa się na 3 m, osobno dla kątów, prędkości i obu czynników. Pełnego treningu na klastrze jeszcze nie uruchomiono.</p><p>Ostatni zakończony v5 latest uzyskał 96,1% na połączonym benchmarku 1–2,2 m, ±10° i składowych vx/vy ±3 m/s. Nie jest to wynik nowej, szerszej domeny. Wyniki i filmy v4/v5 zachowano w etapie 11.</p><p><a href="#e12">Etap 12: profil v6, walidacja i start →</a> · <a href="#e11">Etap 11: wyniki i cztery filmy v4/v5</a></p></section>'''
