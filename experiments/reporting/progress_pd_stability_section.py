"""Report gain ablations, integration checks and the prepared v4 training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/pd_stability_v4'
BASE='../artifacts/pd_stability_v4/'

def build(table):
    s=json.loads((OUT/'summary.json').read_text());full=[];screen=[];replay=[];refine=[]
    variants=['original','kd_075','calf_kd_half','implicit_5ms','implicit_2p5ms','implicit_1p25ms']
    for v in variants:
        selected=[next(r for r in s['full'] if r['variant']==v and r['controller']==n) for n in ['zero','best','latest']]
        full.append([v]+[f'{r["successes"]}/68' for r in selected]+[' / '.join(f'{r["nominal_successes"]}/4' for r in selected)])
    for r in s['screen']:
        screen.append([r['variant']]+[f'{r["controllers"][n]["successes"]}/12; nom. {r["controllers"][n]["nominal_successes"]}/4' for n in ['zero','best','latest']])
    for v in s['variants']:
        rr=[r for r in s['replay'] if r['variant']==v]
        replay.append([v,f'{100*sum(r["calf_saturation_fraction"] for r in rr)/len(rr):.2f}%',f'{sum(r["tracking_rms_deg"] for r in rr)/len(rr):.2f}°'])
    for r in s['refinement']:
        refine.append([r['controller'],f'{r["classification_changes"]}/{r["count"]}',f'{r["median_max_rp_difference_at_respective_contact_deg"]:.4f}°',f'{r["max_rp_difference_at_respective_contact_deg"]:.3f}°'])
    return f'''<section id="e7" class="stage"><div class="stage-label">Etap 07 / mniejsze gainy, kontrola całkowania i przygotowanie v4</div>
<h2>Mniejsze Kd usuwa oscylacje, lecz do kolejnego treningu zachowujemy PD i poprawiamy integrator.</h2>
<p class="lead">Wykonano {s['new_full_episodes']} nowych pełnych epizodów porównania, {s['screen_episodes']} epizody screeningu oraz {s['replay_count']} krótkich replayów. Dodatkowo wykorzystano 204 epizody oryginalnego PD z Etapu 6. Redukcja Kd i implicitfast likwidują naprzemienne nasycenie w badanych replayach. Obniżanie nastaw nie daje wyraźnego zysku lądowań istniejących polityk, więc v4 pozostawia gainy606 i używa implicitfast z krokiem 2,5 ms.</p>
<h3>1. Rozdzielenie Kp, Kd i integratora</h3>
<p>W 60 replayach wykorzystano te same zapisane pozycje zadane best/latest dla h=2,5 m i startu nominalnego albo roll=+30°. Każdy wariant odtwarza pierwsze 0,3 s bez ponownego wywoływania polityki. Porównano Kp×0,5; Kd×0,75/0,5/0,25/0,1; oba gainy ×0,5/0,25; samo Kd łydek ×0,5/0,25 oraz różne kroki i integratory. Oryginalne joint-side Kp: hip 67,4, thigh 75,4, calf 34,8, spine 73,5; Kd: 1,4 / 4,8 / 7,3 / 5,2. Warianty skalują jednocześnie wszystkie odpowiednie aktuatory; home i limity momentów pozostają wspólne.</p>
{table(['Wariant','Nasycenie łydek w replayu','RMS śledzenia w replayu'],replay,'Średnia po czterech replayach, próbki t>0,05 s; nasycenie oznacza ≥95% limitu momentu silnika')}
<p>Samo Kp×0,5 zmniejsza, ale nie usuwa oscylacji: średnie nasycenie łydek wynosi około 22,3% zamiast 94,6%. Już Kd×0,75 usuwa je w badanym fragmencie. Jest to zgodne z problemem dyskretyzacji tłumienia aktuatora: Euler w MuJoCo jest semi-implicit i traktuje tłumienie stawu inaczej niż siły zależne od prędkości w aktuatorach. Integrator implicitfast uwzględnia pochodne takich sił. To mechanizm opisany w <a href="https://mujoco.readthedocs.io/en/stable/computation.html">dokumentacji MuJoCo</a>, a wyniki replayu potwierdzają jego znaczenie w tym modelu. Nie nazywamy całego integratora Euler „jawnie całkowaną fizyką”.</p>
<figure class="chart"><img src="{BASE}gain_replay.svg" alt="Moment i pozycja łydki przy mniejszym Kd i stabilniejszym integratorze"><figcaption>Identyczne komendy latest, h=2,5 m, roll=+30°. Brak oscylacji nie oznacza jeszcze najlepszego podparcia podczas kontaktu.</figcaption></figure>
<h3>2. Dlaczego nie obniżyć po prostu wszystkich gainów?</h3>
{table(['Wariant','PD / zero','Best','Latest'],screen,'Screening: cztery nominalne wysokości i osiem przechyłów ±10/30° na 2,5 m; wynik /12 oraz nominalne /4')}
<p>Duże redukcje osłabiają lądowanie. Kp×0,5 nie przechodzi żadnej z czterech prób nominalnych PD; Kd×0,25 również daje 0/4 nominalnie. Łagodniejsze redukcje Kd zachowują nominalne lądowania, lecz nie wykazują jednoznacznej przewagi w pełnej siatce. To test istniejących wag na zmienionym regulatorze — nie ocena tego, czego można nauczyć politykę od nowa dla innych gainów.</p>
<h3>3. Pełna siatka: 68 stanów na kontroler i wariant</h3>
{table(['Wariant','PD / zero','Best','Latest','Nominalne: PD / best / latest'],full,'h=1/1,5/2/2,5 m; roll albo pitch ±3/10/20/30° oraz nominalnie, v_xy=0')}
<figure class="chart"><img src="{BASE}comparison.svg" alt="Porównanie nasycenia w replayu i sukcesów pełnych epizodów"><figcaption>Każdy wariant pełny obejmuje 204 epizody. Powtórzenia screeningu i pełnej siatki nie są niezależnymi losowaniami. Nie sumujemy ich do estymaty prawdopodobieństwa sukcesu.</figcaption></figure>
<p>Dla implicitfast 2,5 ms best ma 45/68, latest 49/68, PD 25/68. Oryginalnie było 45/68, 44/68 i 23/68. Zachowano wszystkie cztery nominalne lądowania każdej metody. To umiarkowana zmiana transferu starych wag, a nie dowód, że nowy trening będzie skuteczniejszy. Główną przyczyną wyboru jest usunięcie silnych oscylacji numerycznych bez osłabienia PD.</p>
<h3>4. Kontrola kroku: 5, 2,5 i 1,25 ms</h3>
<p>Implicitfast eliminuje badane naprzemienne nasycenie przy wszystkich trzech krokach. W pełnych epizodach klasyfikacja nie jest całkowicie niezależna od dt. Poniższa tabela porównuje 2,5 i 1,25 ms: orientację porównano przed własnym pierwszym kontaktem każdej próby, a nie w identycznym czasie fizycznym.</p>
{table(['Metoda','Zmiana sukces/porażka','Mediana różnicy roll/pitch','Maks. różnica roll/pitch'],refine)}
<p>Różnice orientacji przed kontaktem są zwykle małe, ale wynik końcowy zmienia się dla 1/68 prób PD, 3/68 best i 1/68 latest. Wybieramy 2,5 ms jako zbadany kompromis kosztu i dokładności; nie deklarujemy ścisłej zbieżności kontaktu ani przeniesienia na hardware. Badanie granicznych lądowań na drobniejszym kroku pozostaje częścią końcowej walidacji artykułu.</p>
<h3>5. Następny trening jest przygotowany</h3>
{table(['Element','V4'],[
['Fizyka','implicitfast; 2,5 ms; 8 podkroków; sterowanie nadal 50 Hz'],
['Regulator / mechanika','Oryginalny PD606 i home; skala akcji 0,6 rad; te same limity i kontakt'],
['Wagi','Świeże; brak wczytania best/latest z v3'],
['Curriculum','Rewizja 27.10: grupy 20/20/20/40, h=1–2,2 m; 25% resetów na 2,2 m; roll/pitch od ±1° do ±10°; vx/vy do ±3 m/s'],
['PPO','std startowe 0,15, LR 3e-5, 10 epok i architektura jak w v3'],
['Budżet','30 015 488 przejść; 64 środowiska × 2048; limit Slurm 12 h'],
['Walidacja','Wysokości 1 / 1,5 / 2 / 2,2 m; kąty ±10° na oś; dotychczasowa bramka nominalna; nowa fizyka'],
['Nowe metryki','Nasycenie i istotne zmiany znaku momentów łydek w locie; diagnostyka bez zmiany nagrody'],
['Uruchomienie','sbatch experiments/test2.sh (albo sbatch test2.sh w experiments)']])}
<p>Rewizja 27.10.2026 na życzenie użytkownika: maksymalna wysokość curriculum i walidacji wynosi teraz 2,2 m, a docelowy roll/pitch ±10° na oś. Curriculum nadal zaczyna od ±1° i rozszerza zakres według wyników; ewaluacja używa pełnych ±10°. Zmienia się więc również domena zadania i wyników zagregowanych nie należy przypisywać wyłącznie integratorowi. Wcześniejsze pomiary w tym etapie pozostają wynikami historycznych testów do 2,5 m, a nie pomiarami dla nowej domeny. std pozostaje 0,15. Nie ma jeszcze treningu do 30–45°.</p>
<p><code>test2.sh</code> wskazuje teraz <code>train_landing_seed606_v4.py</code>. Dotychczasowa obsługa Condy i EGL jest zachowana. Odrzucane jest wznowienie checkpointu z innym SHA profilu fizyki. Modele będą w <code>runs/soft_landing/landing_seed606_stable_v4/seed606_stable_v4_JOBID/models/</code>. Trening na klastrze nie został uruchomiony w ramach tej pracy.</p>
<h3>6. Zgodność PPO i search oraz wykonane sprawdzenia</h3>
<p>Nowy wspólny JSON zapisuje integrator, timestep i solver. PPO sprawdza timestep i SHA pliku; search CPU/GPU przyjmuje <code>--physics-profile …/landing_physics_v4.json</code>, a workery CPU dziedziczą ustawienie. Bez jawnego profilu search nadal używa poprzedniej fizyki. Przy optymalizacji search sam dobiera gainy/home; wspólny profil ujednolica integrację, a porównanie PD606 wymaga tych samych nastaw.</p>
<p>Przed rewizją domeny wersja przeszła 31 testów regresyjnych. Po zmianie wysokości i kąta wykonano pięć testów v4, w tym rzeczywiste resety sprawdzające limit 2,2 m i curriculum kąta 1→10°. Test CPU search–PPO sprawdza parametry modelu, kroki 50 Hz, okna 10/50 ms i zgodność qpos/qvel przez kontakt. Rzeczywisty smoke wcześniejszej domeny v4 przez test2.sh wykonał 512 przejść i jedną finalizację epizodu; nowe wskaźniki nasycenia i istotnych odwróceń momentu łydki były zerowe. Ten mały smoke potwierdza uruchomienie, nie skuteczność nauczonej polityki.</p>
<p>Lokalna instalacja nie zawiera mujoco-mjx, więc wykonanie search GPU na nowym profilu nie zostało tu sprawdzone. Implementacja ładuje ten sam JSON, a cache uwzględnia integrator i timestep; zgodność numeryczna CPU/MJX wymaga testu w środowisku klastra. PPO v4 używa fizyki CPU MuJoCo i GPU do uczenia sieci — ta ścieżka nie wymaga MJX.</p>
<p><a href="{BASE}summary.json">Wyniki pełne, screening i replaye</a> · <a href="{BASE}comparison.csv">Tabela CSV</a> · <a href="{BASE}training_profile.json">Rozwiązany profil treningu</a> · <a href="{BASE}training_smoke.log">Log krótkiego treningu</a> · <a href="../../experiments/landing_seed606_v4_README.md">Instrukcja uruchomienia i odtworzenia</a></p></section>'''
