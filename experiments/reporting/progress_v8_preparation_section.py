"""V8 plan and checkpoint-by-checkpoint empirical operating envelope."""
import html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def build(table):
    data=json.loads((ROOT/'docs/report/report-data.json').read_text())
    result='''<section id="e16" class="stage"><div class="stage-label">Etap 16 · v8</div><h2>Najpierw wysokość, potem sama prędkość</h2>
<p class="lead">V8 rozdziela pytania badawcze: teraz uczymy lądowania z prędkością poziomą, bez początkowego roll/pitch. Osobny trening kątów będzie kolejnym eksperymentem; nie dokładamy go automatycznie do v8.</p>
<h3>Co dokładnie zmieniono</h3><p>Dotychczas każdy worker od początku losował wysokości 1–3 m, a grupy nominalna, kątowa, prędkościowa i łączona konkurowały o próbki. Teraz worker zaczyna od 1 m i zerowej prędkości. Kolejne granice wysokości to 1,5; 2; 2,5 i 3 m. Po przejściu wszystkich pięciu poziomów zaczyna prędkości 0,5; 1; 2; 3; 4 i 5 m/s. Wszystkie kierunki poziome są dozwolone; limit dotyczy modułu, a nie każdej składowej.</p>
<p>Każdy worker awansuje niezależnie po pełnym ruchomym oknie 64 prób na aktualnej granicy, jeżeli sukces przekracza 80%: minimum 52/64, czyli 81,25%. Nie wystarcza wysoka średnia z łatwych replayów. Połowa resetów to próba graniczna: w fazie wysokości aktualna wysokość i v=0, w fazie prędkości dokładnie 3 m i aktualny moduł prędkości. Pozostałe próby utrwalają wcześniejsze wysokości; w fazie prędkości korzystają z równomiernego pola dysku do aktualnej prędkości. Prędkość graniczna ma losowy kierunek.</p>
<p>Etapy są monotoniczne: po awansie nie ma cofania ani resetowania wag. Różne workery mogą być na innych etapach. Uśredniony numer etapu w logu nie jest wspólną certyfikacją polityki. Potwierdzenie skuteczności pochodzi z osobnej ewaluacji, a nie ze skuteczności lokalnego curriculum. Pozostają stałe PD606, implicitfast 2,5 ms, residual action scale 0,6 rad, kryteria bezpieczeństwa, nagrody oraz PPO z v7. Pomiar prędkości ma nadal σ=0,1 m/s na składową i opóźnienie 20 ms. Start od nowych wag, 30 015 488 przejść.</p>
<h3>Jak raportujemy „powyżej 80%”</h3><p>Na każdym zaplanowanym sprawdzeniu polityka i zerowa akcja przechodzą tę samą siatkę: wysokości 1; 1,5; 2; 2,5; 3 m oraz dokładne prędkości 0; 0,5; 1; 2; 3; 4; 5 m/s. Dla każdej komórki i każdego kontrolera jest 64 epizodów, z losowym kierunkiem i bez początkowego kąta, pionowej prędkości i spinu. Pełna ewaluacja obejmuje 4480 epizodów i następuje co 4 194 304 przejścia oraz na końcu. Jest droższa niż dawne sprawdzenie tylko na 3 m.</p>
<p>Raport pokaże sukcesy/liczbę prób, procent, 95-procentowy przedział Wilsona oraz krok pierwszego przekroczenia progu. Oddzielnie zapisze wynik ostatniego sprawdzenia. Jednorazowe przekroczenie progu nie oznacza, że końcowy model utrzymał ten wynik. Granica dla każdej wysokości to największa kolejna prędkość z siatki, przy której wszystkie niższe badane prędkości również przekroczyły 80%. Izolowany sukces przy 4 m/s nie przykryje niepowodzenia przy 3 m/s.</p>
<p>Próg empiryczny 52/64 nie oznacza jeszcze, że dolna granica przedziału ufności przekroczyła 80%; raport rozróżnia te dwa warunki. Siatka nie stanowi gwarancji dla wszystkich pośrednich wysokości lub kierunków. Te same seedy ewaluacji służą wyborowi checkpointu, więc końcowe twierdzenia artykułu wymagają nowego held-out zestawu i kolejnych seedów treningu.</p>
<p>Best jest wybierany dopiero, gdy wszystkie wysokości nominalne przekraczają 80%, a następnie według najgorszej średniej po prędkościach na jednej wysokości, z rozstrzyganiem remisu średnią całej domeny. Latest zachowuje ostatnią ewaluację. Sama nazwa best nie oznacza przekroczenia 80% w całej siatce.</p>
<h3>Uruchomienie i późniejsze wczytanie wyników</h3><pre><code>sbatch experiments/test2.sh
rlx/bin/python -m experiments.analysis.summarize_v8_training --run-dir runs/soft_landing/landing_seed606_velocity_only_v8/NAZWA_RUNU
rlx/bin/python -m experiments.reporting.build_progress_report</code></pre>
<p>Historia progów jest zapisywana automatycznie w landing_evaluation/boundary_history.json, pełne epizody w CSV. Na razie v8 jest przygotowaniem treningu; próba techniczna nie jest wynikiem naukowym i nie wypełnia tabeli skuteczności.</p>'''
    results=data.get('v8_results')
    if results:
        result+='<h3>Ostatnia pobrana ewaluacja v8</h3><p>Krok: '+str(results['latest']['step'])+'. Wyniki dotyczą tej ewaluacji; sprawdź osobno best/latest.</p>'
        rows=[]
        for c in results['latest']['cells']:
            if c['controller']!='policy':continue
            lo,hi=c['wilson95'];key=f'h_{c["height"]:g}_v_{c["speed"]:g}'
            first=results['first_pass'].get(key,{})
            rows.append([f'{c["height"]:g} m',f'{c["speed"]:g} m/s',f'{c["successes"]}/{c["episodes"]}',f'{100*c["success_rate"]:.1f}%',f'{100*lo:.1f}–{100*hi:.1f}%',str(first.get('step','—'))])
        result+=table(['Wysokość','Moduł prędkości','Sukcesy','Skuteczność','95% Wilson','Pierwsze >80%: krok'],rows)
        checkpoint_rows=[]
        for name,metrics in results.get('checkpoints',{}).items():
            step=metrics.get('evaluation_nr_env_steps')
            evaluation=next((e for e in results['evaluations'] if e['step']==step),None)
            if evaluation:
                for h,v in evaluation['contiguous_speed_boundary'].items():
                    checkpoint_rows.append([name,str(step),h+' m','brak' if v is None else f'{v:g} m/s'])
        if checkpoint_rows:
            result+='<h4>Końcowe checkpointy: utrzymana granica >80%</h4>'+table(['Model','Krok ewaluacji','Wysokość','Kolejna prędkość >80%'],checkpoint_rows)
        result+='<h4>Granica utrzymana w ostatniej ewaluacji</h4>'+table(['Wysokość','Największa kolejna prędkość >80%'],[[h+' m','brak' if v is None else f'{v:g} m/s'] for h,v in results['latest']['contiguous_speed_boundary'].items()])
    return result+'</section>'

def overview():
    data=json.loads((ROOT/'docs/report/report-data.json').read_text())
    if data.get('v8_results'):
        step=data['v8_results']['latest']['step']
        return f'<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan</div><h2>V8: wyniki samych prędkości</h2><p>Ostatnia pobrana ewaluacja: krok {step}. Granice skuteczności, pierwsze przekroczenia progu i końcowe best/latest opisano w etapie 16.</p><p><a href="#e16">Wyniki i warunki eksperymentu</a></p></section>'
    return '<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan</div><h2>Przygotowany v8: wysokość → prędkość</h2><p>Same prędkości, zerowy początkowy przechył. Awans curriculum i niezależna ewaluacja granic >80% są rozdzielone. Nie ma jeszcze wyniku pełnego treningu v8.</p><p><a href="#e16">Plan, progi i sposób raportowania</a></p></section>'
