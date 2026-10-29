"""Completed velocity-aware v5: paired results and scope of scientific claims."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v5_completed_9379363_2026-10-28'
BASE='../artifacts/v5_completed_9379363_2026-10-28/'
def build(table,videos):
 s=json.loads((OUT/'test_summary.json').read_text());p=json.loads((OUT/'parsed.json').read_text());meta=json.loads((OUT/'checkpoints.json').read_text());movies=json.loads((OUT/'videos.json').read_text());last=p['rows'][-1]['metrics']
 names=['zero','v4','v5_best','v5_latest']
 def stat(kind,n):return next(r for r in s['groups'] if r['kind']==kind and r['controller']==n)
 def score(r):return f'{r["successes"]}/{r["count"]} · {100*r["rate"]:.1f}%'
 kinds=[('nominal','Nominalnie, wysokość co 0,1 m'),('random_angles','Same kąty ±10°'),('random_velocity','Same vx/vy ±3 m/s'),('random_combined','Kąty ±10° i vx/vy ±3 m/s'),('limited_combined','Kąty ±10° i moduł v ≤1 m/s'),('angle_grid','Siatka kątów ±3/10/20/30° (20/30° poza treningiem)')]
 groups=[[label]+[score(stat(k,n)) for n in names] for k,label in kinds]
 hs=[]
 for h in [1.,1.5,2.,2.2]:
  rr=[f'{h:g} m']
  for n in names:
   r=next(r for r in s['by_height'] if r['controller']==n and r['kind']=='random_combined' and r['height']==h);rr.append(f'{r["successes"]}/{r["count"]} · {100*r["successes"]/r["count"]:.1f}%')
  hs.append(rr)
 speeds=[]
 for speed in [.5,1,1.5,2,2.5,3]:
  rr=[f'{speed:g} m/s']
  for n in names:
   r=next(r for r in s['speed_grid'] if r['controller']==n and r['speed']==speed);rr.append(f'{r["successes"]}/{r["count"]}')
  speeds.append(rr)
 bins=[]
 for lo,hi in [(0,1),(1,2),(2,3),(3,4.25)]:
  rr=[f'[{lo:g}; {hi:g}) m/s']
  for n in names:
   r=next(r for r in s['combined_speed_bins'] if r['controller']==n and r['min_inclusive']==lo);rr.append(f'{r["successes"]}/{r["count"]}')
  bins.append(rr)
 failed=[[f'{r["case"]["height"]:g} m',f'{r["speed"]:.2f} m/s',f'{r["case"]["roll"]:+.1f}°',f'{r["case"]["pitch"]:+.1f}°',r['reason']] for r in s['latest_combined_failures']]
 best=stat('random_combined','v5_best');latest=stat('random_combined','v5_latest');v4=stat('random_combined','v4');probe=s['zeroed_probe'];paired=s['paired_comparisons']['v4_vs_v5_best']
 checkpoint=[]
 for n in ['best','latest']:
  m=meta[n]['metrics'];checkpoint.append([n,f'{m["nr_env_steps"]:,}'.replace(',',' '),f'{100*m["eval/landing_success_rate"]:.2f}%',f'{100*m["eval/worst_height_success_rate"]:.2f}%',meta[n]['sha256'][:16]])
 failure_rows=[[n]+[str(s['failures'][n].get(reason,0)) for reason in ['disallowed_contact','body_acceleration','unresolved']]+[str(sum(s['failures'][n].values()))] for n in names]
 films=[dict(title='Prędkość 3 m/s do przodu: v4 i v5 latest',src=movies[0]['src'],note='Cztery wysokości: 1,0; 1,5; 2,0; 2,2 m. Roll=pitch=0. Stały przypadek bez selekcji po wyniku; film 4× wolniej.'),dict(title='Kąty i prędkości: v4 i v5 latest',src=movies[1]['src'],note='Pierwszy zapisany przypadek random_combined na każdej wysokości. Dobór przypadków nie zależy od ich wyniku.')]
 lateral_html=''
 lateral_path=OUT/'lateral_videos.json'
 if lateral_path.exists():
  lateral=json.loads(lateral_path.read_text())
  lateral_films=[dict(title='Wyłącznie w lewo: vy=+3 m/s — v4 i v5 latest',src=lateral[0]['src'],note='vx=0, roll=pitch=0; wysokości 1,0; 1,5; 2,0 i 2,2 m. V4 po lewej, v5 latest po prawej; film 4× wolniej.'),dict(title='Wyłącznie w prawo: vy=−3 m/s — v4 i v5 latest',src=lateral[1]['src'],note='vx=0, roll=pitch=0; te same cztery wysokości i kryteria. Przypadki wybrane przed sprawdzeniem wyniku.')]
  lateral_html='<h3>Filmy: sama prędkość boczna, bez przechyłu</h3><p>Dwa oddzielne kierunki: w lewo (+y) i w prawo (−y) przy początkowym yaw=0. Jedynym zaburzeniem jest prędkość boczna 3 m/s; prędkość wzdłużna oraz roll i pitch są zerowe. Na każdym filmie pokazano cztery wysokości, a wyniki i hashe wszystkich 16 odtworzeń sprawdzono z benchmarkiem.</p>'+videos(lateral_films)+'<p><a href="'+BASE+'lateral_videos.json">Dokładne warunki, znaczniki czasu i wyniki filmów bocznych</a></p>'
 return f'''<section id="e11" class="stage"><div class="stage-label">Etap 11 / 28.10.2026 / zakończony v5 · zadanie 9379363</div>
<h2>Znajomość prędkości zmienia wynik. Sprawdzamy to poza walidacją treningową.</h2>
<p class="lead">Trening (seed=1; 606 oznacza źródłowy regulator) zakończył planowane 30 015 488 przejść. Na identycznych 256 niezależnych scenariuszach z połączonymi zaburzeniami v5 best uzyskał {score(best)}, latest {score(latest)}, a wcześniejszy v4 {score(v4)}. Wysokości: 1–2,2 m; roll i pitch niezależnie ±10°; vx i vy niezależnie ±3 m/s. To wynik symulacji z idealnym pomiarem prędkości, jednego seeda treningowego.</p>
<h3>Co faktycznie zostało wytrenowane?</h3>
<p>Checkpointy potwierdzają tryb heading_xy: dwa dodatkowe wejścia aktora (łącznie 80), prędkość wzdłużna i boczna w poziomej ramie kierunku bazy, skala /4 m/s. Critic ma 94 wejścia. Zachowano PD606, 13 akcji, skalę resztkowej pozycji 0,6 rad, implicitfast 2,5 ms i sterowanie 50 Hz, nagrodę oraz domenę v4. Wagi startowały od nowa. Nie trenowano dobierania Kd na początku epizodu.</p>
<p>W logu jest {len(p['rows'])} agregatów, bez Traceback, CUDA OOM i wpisów NaN. Pełna trudność kątów pojawia się przy 9 961 472 przejściach, prędkości przy 12 320 768. Końcowy rollout: nominalnie i same kąty 100%, sama prędkość 97,6%, połączone zaburzenia 96,1%. Końcowe std=0,0758, przybliżone KL=0,0246, clip fraction=26,8%, explained variance krytyka=0,962. Te wskaźniki nie sygnalizują przerwania uczenia, ale same nie dowodzą optymalności. To ruchomy rozkład treningowy; główne porównanie poniżej pochodzi z ustalonego zestawu testowego.</p>
<figure class="chart"><img src="{BASE}learning_curves.svg" alt="Krzywe uczenia v5, curriculum i porównanie walidacji z v4"><figcaption>Trening od ok. 12:59 do 19:55, 27 października. Dolny lewy wykres pokazuje walidację używaną przy wyborze modelu; nie jest ona niezależnym testem.</figcaption></figure>
<h3>Best a latest — różne wagi, mała różnica skuteczności</h3>
{table(['Checkpoint','Przejścia','Łączne zaburzenia — walidacja','Najgorsza wysokość — walidacja','SHA256 pliku (prefiks)'],checkpoint)}
<p>Best zapisano przy 29 360 128 przejściach, latest przy 30 015 488. Hash przywróconych tablic polityki jest różny. Best wybrano na walidacji podczas treningu; niezależny test służy ocenie tego wyboru, nie ponownemu strojeniu na tych samych próbach. Na benchmarku latest wygrywa 9 dodatkowymi próbami i nie traci żadnego sukcesu best: 237 wspólnych sukcesów, 9 tylko latest, 10 wspólnych porażek. Do dalszych lokalnych analiz latest jest więc praktycznym punktem odniesienia. Nie zmieniamy wstecz kryterium wyboru best: mały zestaw walidacyjny i inny zestaw testowy mogą inaczej porządkować bliskie checkpointy. Potwierdzenie wyboru wymaga świeżego testu, a nie kolejnego strojenia na tym benchmarku.</p>
<h3>Wspólny test: 925 scenariuszy na metodę</h3>
<p>Wykonano 1850 nowych epizodów v5 (925 best i 925 latest). Do porównania użyto archiwalnych 1850 epizodów PD i v4, po dokładnym odtworzeniu 32 prób obu baz. W każdej z 925 par potwierdzono identyczny hash scenariusza. Ten sam stan początkowy, fizyka, PD i kryteria bezpieczeństwa; deterministyczna średnia polityki. Brak randomizacji parametrów fizyki w tym teście. Jest to zamrożony benchmark z poprzedniego etapu, oceniany już dla v4; nie przedstawiamy go jako całkowicie nowego, nigdy wcześniej oglądanego testu końcowego całego projektu.</p>
{table(['Warunki','PD / zero','PPO v4','PPO v5 best','PPO v5 latest'],groups)}
<p>Losowa pełna domena: dla best 95% przedział Wilsona wynosi {100*best['ci95'][0]:.1f}–{100*best['ci95'][1]:.1f}%, dla latest {100*latest['ci95'][0]:.1f}–{100*latest['ci95'][1]:.1f}%. Przedziały dotyczą próbkowania scenariuszy, nie zmienności między seedami treningu. Punkty siatki i testy nominalne są deterministyczne. Przy losowaniu vx/vy osobno ±3 m/s moduł może osiągnąć około 4,24 m/s — nie należy opisywać całej domeny jako modułu ≤3 m/s.</p>
<h3>Wpływ wysokości przy kątach i prędkości jednocześnie</h3>
{table(['Wysokość','PD / zero','PPO v4','PPO v5 best','PPO v5 latest'],hs)}
<p>Na każdą wysokość przypadają 64 niezależne przypadki połączone. Latest ma najniższy wynik przy 1 m: 58/64 (90,6%), wobec 62/64 (96,9%) przy 2,2 m. Nie ma więc prostego trendu „wyżej zawsze trudniej”. Krótszy czas na reakcję przy niskim zrzucie jest możliwym wyjaśnieniem, ale ta tabela sama nie dowodzi mechanizmu. Porównanie jest sparowane również względem starszej polityki: {paired['b_only']} prób best kończy sukcesem tam, gdzie v4 zawiódł; {paired['a_only']} ma sytuację odwrotną. Próg 70–80% należy rozpatrywać w zadanej domenie i przy obecnej definicji sukcesu, a nie jako gwarancję dowolnego lądowania.</p>
<h3>Gdzie przebiega granica prędkości?</h3>
<figure class="chart"><img src="{BASE}speed_envelope.svg" alt="Sukces PD, v4, best i latest w funkcji modułu prędkości i wysokości"><figcaption>Każdy punkt: osiem kierunków, roll=pitch=0. To siatka rozdzielająca sam wpływ prędkości; nie zastępuje testów połączonych.</figcaption></figure>
{table(['Moduł prędkości','PD / zero','PPO v4','PPO v5 best','PPO v5 latest'],speeds)}
<h3>Prędkość w połączonych warunkach — również powyżej 3 m/s</h3>
{table(['Przedział modułu prędkości','PD / zero','PPO v4','PPO v5 best','PPO v5 latest'],bins)}
<p>Podział pochodzi z tych samych 256 prób połączonych. Nie zmienia wag rozkładu ani głównej metryki; ujawnia różnicę między limitem składowej i modułu. Małe liczności w skrajnych przedziałach ograniczają precyzję.</p>
<p>Latest zaliczył wszystkie 205/205 losowych prób połączonych z modułem prędkości poniżej 3 m/s. Wszystkie jego 10 porażek w tej grupie przypada na moduł ponad 3 m/s; w przedziale 3–4,25 m/s wynik wynosi 41/51 (80,4%). Nie jest to gwarancja ciągłej granicy 3 m/s, ale wyraźnie lokalizuje trudniejszy fragment badanej domeny.</p>
<h3>Czy polityka rzeczywiście korzysta z prędkości?</h3>
<p>Dodatkowy test zachowuje wagi best, fizykę, pozostałe obserwacje i te same 256 przypadków połączonych, ale w inferencji zeruje oba kanały prędkości. Wynik: {probe['successes']}/{probe['count']} ({100*probe['successes']/probe['count']:.1f}%), wobec {score(best)} z poprawnymi kanałami. Wskazuje to na silną zależność wyuczonego sterowania od dostarczanej informacji o ruchu.</p>
<p>Jest to interwencja podczas testu, która przekazuje sieci błędną informację „prędkość zero”. Nie jest treningiem kontrolnym masked i nie dowodzi, że całą różnicę v4–v5 powodują wyłącznie dwa kanały. Do izolowanej ablacji potrzebne są świeże treningi heading_xy i masked, przy identycznej architekturze 80 wejść i kilku sparowanych seedach.</p>
<h3>Bezpieczeństwo i pozostałe porażki</h3>
{table(['Metoda','Niedozwolony kontakt','Przyspieszenie ciała','Brak rozstrzygnięcia','Wszystkie porażki / 925'],failure_rows)}
<p>Tabela pokazuje wybrane przyczyny oraz całkowitą liczbę porażek; pełny słownik przyczyn znajduje się w JSON. Dawna niestabilność napędów nie wróciła: w 229 agregatach treningu oba wskaźniki przedkontaktowe łydki (nasycenie i odwrócenia znaku) wynoszą zero. Maksima w nowym teście zapisano osobno. Nie oznacza to, że wszystkie napędy nigdy nie osiągają limitu momentu.</p>
<h4>Wszystkie porażki latest w 256 próbach połączonych</h4>
{table(['Wysokość','Moduł prędkości','Roll','Pitch','Przyczyna'],failed)}
<p>Pierwszy niedozwolony kontakt w 10 połączonych porażkach latest: 6 razy obudowa uda (FR/FL), 3 razy rear_mesh i raz górna łydka RL. Ograniczenie nie sprowadza się więc do zbyt miękkiego kontaktu dolnej części łydki. Wcześniejsze zakończenie nieudanych prób ogranicza interpretację porównań sił: mniejszy zarejestrowany pik nie musi oznaczać bezpieczniejszego lądowania.</p>
<p>Sukces wymaga kontaktu stopy, co najmniej 2 s od pierwszego kontaktu i ostatnich 0,5 s stabilności: prędkość liniowa ≤0,5 m/s, kątowa ≤2 rad/s, kontakt z podłożem i brak kontaktu korpusu. Przez próbę obowiązują wspólne limity bezpieczeństwa. Nadal fizycznie dozwolona jest dolna 1/3 łydki; górna 2/3 i inne niedozwolone kontakty kończą próbę. Metryka nie jest obietnicą długotrwałego chodu po lądowaniu ani wynikiem na sprzęcie.</p>
<h3>Filmy: v4 i v5 w identycznych warunkach</h3>
{videos(films)}
{lateral_html}
<p>W odtwarzaczach: v4 po lewej, v5 latest po prawej. Zachowano też wcześniejsze wersje porównania z best: <a href="media/etap11-predkosc-3ms.mp4">3 m/s — best</a> i <a href="media/etap11-pierwsze-proby-laczone.mp4">warunki połączone — best</a> (z zachowaną porażką best na 2,2 m).</p>
<p>Filmy wybierają z góry określone przypadki: 3 m/s do przodu bez przechyłu oraz pierwszy archiwalny losowy przypadek połączony na każdej wysokości. To ilustracje, nie estymator skuteczności. Każde rozstrzygnięcie i hash filmu sprawdzono z wynikami testu. Po zakończeniu epizodu dany panel zatrzymuje obraz.</p>
<h3>Wnioski do artykułu i następne kroki</h3>
<p>Najsilniejszy wynik to skuteczne lądowanie z dużą prędkością poziomą przy dostępnej informacji o ruchu. PD pozostaje punktem startowym konstrukcji polityki. W publikacji należy pokazać mapy względem wysokości, kierunku i modułu prędkości, a kąty pozostawić jako osobny czynnik, bez budowania narracji o wyższości PPO nad regulatorem.</p>
<p>Przed zwiększaniem zakresu warto utrwalić ten wynik: kilka seedów heading_xy i masked; końcowy, zamrożony zestaw testowy, którego nie użyjemy do doboru hiperparametrów; następnie zmierzony model błędu i opóźnienia OptiTrack oraz randomizacja fizyki. Obecny wynik dotyczy idealnego sygnału (błąd w logu = 0), znanej płaskiej powierzchni i tej samej symulacji. Nie rozszerza automatycznie wniosków na rzeczywisty robot, 30° przechyłu ani wysokości ponad 2,2 m.</p>
<p><a href="{BASE}test_summary.json">Podsumowanie i przedziały</a> · <a href="{BASE}paired_results.csv">Wszystkie sparowane wyniki CSV</a> · <a href="{BASE}test_protocol.json">Protokół i 925 scenariuszy</a> · <a href="{BASE}parsed.json">Cały przebieg treningu</a> · <a href="{BASE}checkpoints.json">Metadane checkpointów</a> · <a href="{BASE}baseline_replay_verification.json">32 odtworzenia baz</a> · <a href="{BASE}velocity_zeroed_probe.json">Test zerowania prędkości</a> · <a href="{BASE}videos.json">Warunki i wyniki filmów</a> · <a href="{BASE}configuration_comparison.json">Zgodność konfiguracji v4/v5</a> · <a href="{BASE}source_hashes.json">Pochodzenie kodu</a></p></section>'''

def overview():
 s=json.loads((OUT/'test_summary.json').read_text())
 def rate(n):
  r=next(r for r in s['groups'] if r['controller']==n and r['kind']=='random_combined');return f'{r["successes"]}/{r["count"]} ({100*r["rate"]:.1f}%)'
 return f'''<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan / zakończony v5 z prędkością · 9379363</div><h2>Polityka z informacją o prędkości przekracza cel w badanej symulacji.</h2><p class="lead">Niezależne, sparowane przypadki: v5 best {rate('v5_best')}, latest {rate('v5_latest')}, v4 {rate('v4')}. Domena połączona: h=1–2,2 m, roll/pitch ±10°, każda składowa vx/vy ±3 m/s. Testy obu checkpointów obejmują po 925 scenariuszy.</p><p>Wynik dotyczy jednego seeda treningu, idealnego pomiaru prędkości i znanej fizyki. Wyzerowanie nowych kanałów w best bez zmiany wag obniża wynik do 54/256 (21,1%); osobny trening masked i powtórzenia po seedach pozostają potrzebne. Dodano mapy prędkości, porównanie wysokości i dwa filmy v4/v5.</p><p><a href="#e11">Etap 11: pełne wyniki i filmy →</a> · <a href="#e10">Konfiguracja treningu v5 i OptiTrack</a> · <a href="#e8">Wcześniejszy v4</a></p></section>'''
