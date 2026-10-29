"""Completed v4: archived log, independent evaluation and matched video montages."""
import json,html,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v4_completed_9376691_2026-10-27'
BASE='../artifacts/v4_completed_9376691_2026-10-27/'

def build(table,videos):
    parsed=json.loads((OUT/'parsed.json').read_text());meta=json.loads((OUT/'checkpoints.json').read_text());s=json.loads((OUT/'test_summary.json').read_text());movie=json.loads((OUT/'videos.json').read_text());last=parsed['rows'][-1]['metrics'];checkpoint=meta['best']['metrics']
    def stat(kind,name):return next(r for r in s['groups'] if r['kind']==kind and r['controller']==name)
    def score(r,ci=False):
        text=f'{r["successes"]}/{r["count"]} · {100*r["rate"]:.1f}%'
        if ci:text+=f' (95% Wilson: {100*r["ci95"][0]:.1f}–{100*r["ci95"][1]:.1f}%)'
        return text
    cluster=[];fresh=[];groups=[]
    for h in [1.,1.5,2.,2.2]:
        tag=f'h_{h:g}'.replace('.','p');cluster.append([f'{h:g} m']+[f'{100*checkpoint[f"eval/{kind}/policy/{tag}/success_rate"]:.1f}%' for kind in ['nominal','angles','velocity','perturbed']])
        row=[f'{h:g} m']
        for name in ['zero','v4']:
            r=next(r for r in s['by_height'] if r['controller']==name and r['kind']=='random_combined' and r['height']==h);row.append(f'{r["successes"]}/{r["count"]} · {100*r["successes"]/r["count"]:.1f}%')
        fresh.append(row)
    for kind,label in [('nominal','Nominalne wysokości 1–2,2 m co 0,1 m'),('random_angles','Losowe same kąty ±10°'),('random_velocity','Losowe same vx/vy ±3 m/s'),('random_combined','Losowe kąty ±10° + vx/vy ±3 m/s'),('limited_combined','Losowe kąty ±10° + moduł prędkości do 1 m/s')]:groups.append([label]+[score(stat(kind,n),kind=='random_combined') for n in ['zero','v4']])
    entries=[];clips=[]
    for vid in movie:
        nominal=vid['label']=='nominalne'
        entries.append(dict(title='Nominalne lądowania: 1,0 / 1,5 / 2,0 / 2,2 m' if nominal else 'Połączone zaburzenia: trzy sukcesy polityki i jej ograniczenie',src=vid['src'],note='PD / zero akcji i PPO na identycznych resetach; best i latest mają identyczne wagi. Odtwarzanie 4× wolniej. '+('Cztery nominalne stany wybrane przed testem.' if nominal else 'Przykłady dobrane po analizie: największy moduł prędkości w sukcesach PPO przy porażce PD dla trzech wysokości oraz pierwsza porażka PPO na 2,2 m. Statystyki obejmują cały zestaw.')))
        for clip in vid['clips']:
            c=clip['case'];outcomes={r['controller']:r for r in clip['outcomes']};t=int(clip['start_seconds']);clips.append([vid['label'],f'{t//60}:{t%60:02d}',f'{c["height"]:g}',f'{c["roll"]:+.1f} / {c["pitch"]:+.1f}',f'{c["vx"]:+.2f} / {c["vy"]:+.2f}']+[('sukces' if outcomes[n]['success'] else outcomes[n]['reason']) for n in ['zero','v4']])
    raw=json.loads((OUT/'test_results.json').read_text())
    speed_rows=[];angle_rows=[]
    for speed in [.5,1,1.5,2,2.5,3]:
        speed_rows.append([f'{speed:g} m/s']+[f'{sum(r["success"] for r in raw if r["controller"]==n and r["case"]["kind"]=="speed_grid" and r["case"]["speed"]==speed)}/32' for n in ['zero','v4']])
    for angle in [3,10,20,30]:
        angle_rows.append([f'±{angle}°', 'w domenie treningu' if angle<=10 else 'poza domeną treningu']+[f'{sum(r["success"] for r in raw if r["controller"]==n and r["case"]["kind"]=="angle_grid" and abs(r["case"]["angle"])==angle)}/16' for n in ['zero','v4']])
    v=last['env_info/curriculum/velocity_difficulty'];a=last['env_info/curriculum/angle_difficulty'];full=stat('random_combined','v4');pd=stat('random_combined','zero');limited=stat('limited_combined','v4')
    failure_table=table(['Przyczyna porażki','PD / zero','PPO best=latest'],[[key,s['failures']['zero'].get(key,0),s['failures']['v4'].get(key,0)] for key in sorted(set(s['failures']['zero'])|set(s['failures']['v4']))],'Zbiorczo po heterogenicznych zestawach; liczby opisują diagnostykę, nie rozkład awarii w zastosowaniu')
    return f'''<section id="e8" class="stage"><div class="stage-label">Etap 08 / 27.10.2026 / zakończony trening 9376691</div>
<h2>V4 po treningu: przechyły ±10° i granica skutecznego lądowania.</h2>
<p class="lead">Trening zakończył 30 015 488 przejść. Końcowa walidacja daje 100% nominalnie i przy samych przechyłach oraz 50,4% przy połączonych zaburzeniach. W niezależnym zestawie połączonym nowa polityka uzyskuje {full['successes']}/{full['count']} ({100*full['rate']:.1f}%), a PD {pd['successes']}/{pd['count']} ({100*pd['rate']:.1f}%). Best i latest zawierają identyczne odczytane wagi polityki.</p>
<h3>Co dokładnie zostało wytrenowane?</h3>
{table(['Element','Potwierdzone w pobranych plikach'],[
['Przebieg','seed606_stable_v4_9376691; jeden seed treningu'],
['Budżet i koniec','30 015 488 przejść; 261 557 epizodów; ostatnie metryki 27.10 o 07:04:02 (czas zapisany w logu)'],
['Dynamika','implicitfast; fizyka 2,5 ms; sterowanie 50 Hz; PD606/home i kontakt bez zmian'],
['Domena','Wysokość bazy 1–2,2 m; roll/pitch docelowo ±10° na oś; vx/vy docelowo ±3 m/s na składową'],
['Curriculum','Osobne poziomy kąta i prędkości, grupy 20/20/20/40; wysokość niezależna od trudności'],
['Best i latest','Oba z kroku 30 015 488; identyczne tablice parametrów polityki, również logstd'],
['Pochodzenie danych','Archiwum zawiera log, dokładne kopie obu modeli, hashe plików i odczytanych parametrów']])}
<p>Pliki .model mają różne hashe archiwów z powodu metadanych i reprezentacji zapisu, lecz odczytane parametry polityki są identyczne. Dlatego nie przedstawiamy ich jako dwóch niezależnych metod ani dwóch seedów. Główne testy oceniają jedną reprezentatywną politykę; dodatkowe 16 replayów latest potwierdza identyczny wynik, czas, akcję maksymalną i metryki kontaktu względem best.</p>
<h3>Przebieg nauki i weryfikacja wcześniejszych oscylacji</h3>
<figure class="chart"><img src="{BASE}learning_curves.svg" alt="Krzywe walidacji, trudności curriculum, sukcesów rolloutów oraz diagnostyki napędów w v4"><figcaption>229 zapisanych rolloutów, 29 ewaluacji. Zacienienie 70–80% oznacza cel pracy. Rollouty są mierzone na bieżącej trudności; walidacja stale używa pełnego zakresu.</figcaption></figure>
<p>Końcowy poziom kąta to {a:.4f}, co odpowiada około {1+9*a:.2f}° docelowego zakresu. Poziom prędkości to {v:.4f}; we wzorze 0,5+2,5d odpowiada około {0.5+2.5*v:.2f} m/s na składową. Są to agregaty lokalnych curricula workerów, a nie identyczny poziom każdego środowiska. Raportowane amplitudy uśrednione po wszystkich grupach są niższe, bo część grup ma daną składową zerową.</p>
<p>Na końcu rollouty mają 100% sukcesów nominalnie i kątowo, około {100*last['rollout/group_velocity_success_rate']:.1f}% dla prędkości i {100*last['rollout/group_combined_success_rate']:.1f}% dla połączenia. Nie jest to osiągnięcie 70–80% na pełnej domenie: prędkość treningowa jest zwykle mniejsza niż stały zakres walidacji.</p>
<p>Oba wskaźniki napędów łydki — udział próbek z momentem ≥95% limitu i istotne odwrócenia znaku — wynoszą zero we wszystkich 229 zapisanych agregatach. Wskazuje to, że wcześniejszy silny wzorzec naprzemiennego nasycenia nie powrócił w tych pomiarach. Nie oznacza braku nasycenia przy kontakcie ani dowodu bezpieczeństwa sprzętu. Końcowe KL≈{last['policy_ratio/approx_kl']:.3f}, clip fraction≈{last['policy_ratio/clip_fraction']:.3f}, std≈{last['policy/std_dev']:.3f}. W logu nie wykryto tracebacku, CUDA OOM ani wpisów NaN.</p>
<h3>Stała walidacja z klastra</h3>
{table(['Wysokość','Nominalnie','Same kąty ±10°','Sama prędkość','Kąty + prędkość'],cluster,'64 epizody na warunek i wysokość; vx/vy U[−3,3] m/s; stały seed walidacyjny 20261023')}
<p>Średnia dla prędkości to 55,1%; dla warunków połączonych 50,4%, a najgorsza wysokość ma 46,9%. Końcowe wagi są jednocześnie best, więc log nie sugeruje, że późniejszy checkpoint pogorszył wcześniej wybraną politykę. Ten zestaw służył selekcji modelu i pozostaje walidacją.</p>
<h3>Nowe testy: rozdzielenie wysokości, kątów i prędkości</h3>
<p>Główny test obejmuje 925 stanów, każdy odtworzony dla PD i PPO: 1850 ocenianych epizodów oraz 16 dodatkowych replayów latest. Hashe początkowego stanu i modelu są zgodne w każdej parze. Użyto nowych losowań, seed protokołu 2026102701. Nie zmieniano regulatora, fizyki ani kryterium sukcesu. Yaw=0, początkowe vz=0, brak dodatkowego zaburzenia prędkości kątowej.</p>
{table(['Warunki','PD / zero akcji','PPO best=latest'],groups)}
<p>Losowy zestaw pełny ma niezależne roll/pitch U[−10°,10°] oraz vx/vy U[−3,3] m/s; moduł prędkości może sięgać około 4,24 m/s. W zestawie ograniczonym moduł jest losowany jednostajnie z [0,1] m/s, a kierunek niezależnie. To nie jest box vx/vy ±1 m/s ani jednostajny rozkład po powierzchni dysku. Ograniczony zestaw zaplanowano na podstawie wcześniejszych obserwacji v3, więc traktujemy go jako eksploracyjne wyznaczanie zakresu działania.</p>
{table(['Wysokość','PD / zero','PPO best=latest'],fresh,'Nowe połączone zaburzenia: 64 stany na wysokość')}
<p>Przedziały opisują niepewność po scenariuszach dla jednego treningu; nie opisują powtarzalności po seedach uczenia. Stała walidacja i nowy zestaw mogą dawać różne odsetki. Nie zastępujemy pełnej domeny ograniczoną tylko dlatego, że osiąga wyższy wynik.</p>
<p>W ograniczonym zestawie uzyskano 128/128 sukcesów; dolna granica 95% przedziału Wilsona wynosi około 97,1% dla tego rozkładu stanów. Każda wysokość ma jednak tylko 32 próby (32/32: dolna granica około 89,3%). Nie są to gwarancje dla każdego stanu w domenie ani wynik wielu seedów uczenia.</p>
<h3>Gdzie leży granica prędkości i kąta?</h3>
{table(['Moduł prędkości','PD / zero','PPO best=latest'],speed_rows,'Każdy wiersz: cztery wysokości × osiem kierunków; roll=pitch=0')}
<p>Polityka przechodzi wszystkie 32 kierunki/wysokości do 1,5 m/s włącznie i 28/32 przy 2 m/s. Przy 2,5 m/s wynik spada do 6/32, a przy 3 m/s do 1/32. Samą większą wysokością nie wyjaśnimy tej granicy; mapa pokazuje wynik osobno na każdej wysokości.</p>
<figure class="chart"><img loading="lazy" src="{BASE}speed_envelope.svg" alt="Skuteczność ośmiu kierunków prędkości na czterech wysokościach"><figcaption>Roll=pitch=0. Każdy punkt oznacza osiem ustalonych kierunków, a nie gwarancję dla wszystkich stanów. Łączenie punktów nie wyznacza ciągłej granicy wykonalności.</figcaption></figure>
<figure class="chart"><img loading="lazy" src="{BASE}angle_map.svg" alt="Mapa lądowań z przechyłów od minus 30 do plus 30 stopni"><figcaption>±20° i ±30° to test poza domeną treningową. Kolumna nominalna jest pokazana na obu osiach, ale liczona tylko raz. W głównych próbach kątowych dodatkowo badano cztery narożniki roll/pitch ±10° na każdej wysokości.</figcaption></figure>
{table(['Moduł kąta','Relacja do treningu','PD / zero','PPO best=latest'],angle_rows,'Cztery wysokości × dwie osie × dwa znaki = 16 prób na wiersz; drugi kąt i prędkości zerowe')}
<p>W czterech narożnikach jednoczesnego roll/pitch ±10° polityka przechodzi 16/16 prób, a PD 3/16. Wyniki 15/16 przy 20° i 9/16 przy 30° pokazują częściową generalizację, lecz nie zastępują testu losowych łączonych zaburzeń ani nie oznaczają treningu do 30°.</p>
<h3>Filmy: ten sam PD i ten sam stan początkowy</h3>
{videos(entries)}
<details><summary>Dokładne warunki i wyniki fragmentów filmów</summary>{table(['Film','Początek','h [m]','Roll / pitch [°]','vx / vy [m/s]','PD','PPO'],clips)}</details>
<h3>Przyczyny porażek i pomiary</h3>{failure_table}
<p>Próby kończą się w momencie rozstrzygnięcia, więc szczyty obciążeń nieudanych epizodów mogą być ucięte. Nie wyciągamy z ich prostego porównania wniosku o mniejszych uszkodzeniach. Surowe wyniki zawierają pierwszy niedozwolony kontakt, czas, parę geomów oraz metryki diagnostyczne napędów.</p>
<h3>Wnioski w ramach artykułu</h3>
<p><strong>Potwierdzone:</strong> polityka na stabilniejszej numerycznie dynamice opanowała badane lądowania nominalne i małe przechyły do docelowych ±10°. Zmierzyliśmy jej zakres działania przy prędkości i łączonych zaburzeniach, a filmy odtwarzają stany z pełnego zestawu danych. PD / zero akcji jest ablacją korekty polityki na wspólnej bazie sterowania.</p>
<p><strong>Ograniczenie:</strong> wynik nowego pełnego zestawu połączonego to {100*full['rate']:.1f}%; próg 70–80% w tej domenie nie jest jeszcze potwierdzony. Wynik {100*limited['rate']:.1f}% w ograniczonym zestawie dotyczy innego rozkładu stanów. Jeden seed treningu i jeden model miękkiego kontaktu nie wystarczają do deklaracji powtarzalności ani sim-to-real.</p>
<p><strong>Porównanie z v3:</strong> zmienił się integrator i timestep, wysokość maksymalna 2,5→2,2 m oraz przechyły ±3→±10°. Odsetek 50,4% kontra wcześniejsze 33,2% na klastrze jest porównaniem różnych protokołów. Nie jest izolowanym dowodem wpływu integratora ani przewagi PPO nad optymalizacją PD. Do analizy przyczynowej potrzeba wspólnego zakresu stanów i kontrolowanych ablacji.</p>
<p><strong>Dalsza decyzja:</strong> zachować fizykę v4 jako jawny profil, wykonać powtórzenia treningu i testy wspólnych warunków. Następny wariant powinien sprawdzać konkretną hipotezę dotyczącą prędkości — informacji dla aktora, spójności eksploracji lub przygotowania kontaktu — zamiast zwiększać jednocześnie wszystkie zakresy. Same przechyły nie są już główną słabością w badanej domenie.</p>
<p><a href="{BASE}parsed.json">229 rolloutów i 29 walidacji</a> · <a href="{BASE}checkpoints.json">Metadane i hashe modeli</a> · <a href="{BASE}test_protocol.json">Nowy protokół</a> · <a href="{BASE}test_results.json">Wszystkie epizody</a> · <a href="{BASE}paired_results.csv">Sparowana tabela CSV</a> · <a href="{BASE}latest_verification.json">Weryfikacja latest</a> · <a href="{BASE}videos.json">Warunki i hashe filmów</a> · <a href="{BASE}source_hashes.json">Hashe archiwalnych źródeł eksperymentu</a></p></section>'''
