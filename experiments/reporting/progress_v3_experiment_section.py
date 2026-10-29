"""Long-form report section for completed checkpoint experiments."""
import html
import json
from pathlib import Path
from experiments.analysis.summarize_seed606_v3_experiments import wilson

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v3_checkpoint_experiments_2026-10-26'


def build(table,videos):
    summary=json.loads((OUT/'summary.json').read_text())
    metadata=json.loads((OUT/'checkpoints.json').read_text())
    video_data=json.loads((OUT/'videos.json').read_text())
    probe=json.loads((OUT/'observability_probe.json').read_text())['summary']
    full=[];limited=[];speeds=[]
    for height in [1.,1.5,2.,2.5]:
        row=[f'{height:g} m'];rr=[f'{height:g} m']
        for controller in ['zero','best','latest']:
            x=next(r for r in summary['heldout'] if r['controller']==controller and r['height']==height)
            row.append(f'{x["successes"]}/64 · {100*x["successes"]/64:.1f}%')
            x=next(r for r in summary['limited_region'] if r['controller']==controller and r['height']==height)
            rr.append(f'{x["successes"]}/32 · {100*x["successes"]/32:.1f}%')
        full.append(row);limited.append(rr)
    for speed in [.5,1.,1.25,1.5,2.,2.5,3.]:
        row=[f'{speed:g} m/s']
        for controller in ['zero','best','latest']:
            selected=[r for r in summary['speed_grid'] if r['controller']==controller and r['speed']==speed]
            k=sum(r['successes'] for r in selected);n=sum(r['count'] for r in selected)
            row.append(f'{k}/{n} · {100*k/n:.1f}%')
        speeds.append(row)
    ci_text=[];region_text=[]
    for controller in ['zero','best','latest']:
        x=summary['per_controller'][controller]['heldout'];lo,hi=wilson(x['successes'],x['count'])
        ci_text.append(f'{controller}: {x["successes"]}/{x["count"]} = {100*x["rate"]:.1f}% (95% Wilson: {100*lo:.1f}–{100*hi:.1f}%)')
        x=summary['per_controller'][controller]['limited_region']
        region_text.append(f'{controller}: {x["successes"]}/{x["count"]} = {100*x["rate"]:.1f}%')
    pairing=summary['pairing']['best_vs_latest']
    region_cis=[]
    for name in ['best','latest']:
        row=next(r for r in summary['limited_region'] if r['controller']==name and r['height']==2.5)
        region_cis.append(f'{name}: {100*row["successes"]/row["count"]:.1f}% (95% Wilson: {100*row["ci95"][0]:.1f}–{100*row["ci95"][1]:.1f}%)')
    video_entries=[]
    for item in video_data:
        c=item['case'];success=item['label']=='policy_improvement'
        video_entries.append(dict(title=('Gdzie PPO pomaga' if success else 'Gdzie PPO nadal przegrywa')+f' · {c["height"]:g} m, |v_xy|={c["speed"]:g} m/s',src='media/etap5-'+item['label']+'.mp4',note=f'Trzy kontrolery w tym samym scenariuszu: kierunek {c["direction"]}°, roll=pitch=0. Obraz zatrzymuje się po rozstrzygnięciu każdej próby. Przykład wybrano z pełnej siatki dla ilustracji; statystyki obejmują wszystkie próby.'))
    from experiments.reporting.progress_v3_comparison import build as comparison_build
    comparison=comparison_build(table)
    from experiments.reporting.progress_joint_range_section import build as joint_range_build
    joint_ranges=joint_range_build(table)
    count=summary['total_episodes']
    return f'''<section id="e5" class="stage"><div class="stage-label">Etap 05 / 26.10.2026 / eksperymenty na pobranych wagach</div>
<h2>Best i latest: gdzie polityka potrafi już lądować?</h2>
<p class="lead">Przeprowadzono {count} ocenianych epizodów: 656 stanów początkowych, każdy odtworzony dla zerowej akcji, best i latest. PD, home, ograniczenia i model fizyki pozostają wspólne. Wyniki pokazują użyteczny zakres małych prędkości oraz silny spadek skuteczności po jego przekroczeniu.</p>
<h3>Pochodzenie modeli i protokół</h3>
{table(['Model','Krok wag','Krok zapisanej ewaluacji'],[[name,f'{metadata[name]["metrics"]["nr_env_steps"]:,}'.replace(',',' '),f'{metadata[name]["metrics"]["evaluation_nr_env_steps"]:,}'.replace(',',' ')] for name in ['best','latest']])}
<p>Wagi odpowiadają treningowi 9326190. Loader zweryfikował identyfikator regulatora, hash presetu i modelu oraz zgodność obserwacji. Archiwalne kopie checkpointów i ich SHA256 zapisano z wynikami. W każdej próbie porównano hash początkowego stanu oraz modelu, a po epizodzie sprawdzono niezmienione gainprm i biasprm regulatora.</p>
<p>Zestaw główny obejmuje 16 nominalnych wysokości co 0,1 m, 192 scenariusze prędkości, 32 scenariusze samych kątów i 256 nowych losowych scenariuszy łączonych na kontroler. Następnie dodano 32 stany przy prędkości 1,25 m/s oraz 128 losowych stanów w ograniczonym obszarze do 1 m/s. Dodatkowe zestawy są oznaczone jako eksploracyjne, wybrane po obejrzeniu siatki. Nowe scenariusze nie były używane do trenowania tych checkpointów; po tej analizie stają się danymi rozwojowymi dla dalszych decyzji.</p>
<h3>1. Nominalne wysokości i same przechyły</h3>
<p>Best, latest i zerowa akcja przechodzą wszystkie 16 nominalnych wysokości 1–2,5 m. W kontrolowanym teście orientacji oba checkpointy przechodzą 32/32 scenariusze, a PD 24/32. Badano cztery wysokości oraz osiem kombinacji: roll lub pitch ±3° osobno oraz cztery narożniki roll/pitch ±3°. To potwierdza zachowanie nominalnego lądowania i odporność na małe przechyły, ale nie jest testem dużej reorientacji w locie.</p>
<h3>2. Granica prędkości bez przechyłu</h3>
{table(['Moduł v_xy','Zerowa akcja','Best','Latest'],speeds,'Każdy wiersz: cztery wysokości × osiem kierunków; roll=pitch=0')}
<figure class="chart"><img src="../artifacts/v3_checkpoint_experiments_2026-10-26/speed_envelope.svg" alt="Skuteczność siatki kierunków zależnie od modułu prędkości i wysokości"><figcaption>Odsetek ośmiu przetestowanych kierunków, nie oszacowanie prawdopodobieństwa po wszystkich możliwych stanach. Punkt 1,25 m/s doprecyzowano po analizie pierwszej siatki.</figcaption></figure>
<p>Do 1 m/s oba checkpointy przechodzą wszystkie zbadane kierunki i wysokości. Przy 1,25 m/s latest przechodzi 32/32, best 28/32; przy 1,5 m/s latest 25/32, best 21/32. Przy 2 m/s obie polityki mają tylko 4/32 sukcesy, a przy 2,5 i 3 m/s nie ma sukcesów. W szczególności przy h=2,5 m obie przegrywają wszystkie osiem kierunków z 2 m/s. Nie wolno z tego wyznaczać dokładnej ciągłej granicy ani zakładać monotoniczności między punktami.</p>
<figure class="chart"><img src="../artifacts/v3_checkpoint_experiments_2026-10-26/direction_map.svg" alt="Sukces i porażka na wysokości 2,5 m zależnie od kierunku i prędkości"><figcaption>Zielony: sukces; bordowy: porażka. Kierunek w światowym XY, yaw=0; 0° oznacza +x, 90° +y. Zdolność nie jest izotropowa — nie wystarcza jedna liczba maksymalnej prędkości.</figcaption></figure>
<h3>3. Nowe losowe stany na pełnym zakresie</h3>
{table(['Wysokość','Zerowa akcja','Best','Latest'],full,'Po 64 nowe scenariusze na wysokość; roll/pitch U[−3°,3°], vx/vy U[−3,3] m/s; seed 2026102601')}
<p>{html.escape('; '.join(ci_text))}.</p>
<p>Best ma punktowo lepszy wynik przy łączonych zaburzeniach, ale latest lepiej radzi sobie w części czysto prędkościowej. W sparowanym porównaniu losowych scenariuszy best wygrywa samodzielnie {pairing['a_only']} prób, latest {pairing['b_only']}; dokładny dwustronny test dla prób rozbieżnych daje p≈{pairing['exact_two_sided_p']:.3f}. To nie rozstrzyga wyraźnej przewagi jednego checkpointu. Przedziały opisują niepewność scenariuszy dla tych wag, a nie zmienność między seedami treningu.</p>
<p>Nowy zestaw jest słabszy od walidacji z klastra, która miała około 33,2%. Różne losowania i wybór checkpointów na stałym zestawie walidacyjnym mogą wpływać na tę różnicę. Nie przypisujemy jej automatycznie przeuczeniu, lecz nie traktujemy wyniku walidacyjnego jako potwierdzonej generalizacji. Pierwotny cel 70–80% przy pełnych zaburzeniach pozostaje nieosiągnięty.</p>
<h3>4. Czy cel jest osiągalny w ograniczonej domenie?</h3>
{table(['Wysokość','Zerowa akcja','Best','Latest'],limited,'Test eksploracyjny: po 32 nowe próby/h, |v_xy| U[0,1] m/s, losowy azymut, roll/pitch ±3°, seed 2026102602')}
<p>{html.escape('; '.join(region_text))}.</p>
<p>Przy 2,5 m: {html.escape('; '.join(region_cis))}. Przedziały są osobne dla każdej komórki i nie stanowią jednoczesnej gwarancji dla całej domeny.</p>
<p>Ten wynik określa mniejszy, praktycznie użyteczny obszar, a nie poprawę na pierwotnym pełnym teście. W tym zestawie losowany jest moduł prędkości do 1 m/s i niezależny kierunek; nie jest to box vx/vy ±1 m/s. Obszar wybrano po screeningu, więc do mocnego twierdzenia o zakresie ≥80% potrzebny jest kolejny zamrożony test, więcej scenariuszy na wysokość i niezależne seedy treningu. Nie zamieniamy post factum tego zakresu w pierwotny cel pracy.</p>
<h3>5. Jak fizycznie kończą się porażki?</h3>
<p>W głównym zestawie best ma 296 porażek, z czego 290 sklasyfikowano jako niedozwolony kontakt, 5 jako base crash i 1 jako przekroczenie acceleration. Najczęstszy pierwszy niedozwolony kontakt dotyczy podłoża i obudowy przedniego uda: FR 81 razy, FL 88. Pojawiają się również górne części łydek i korpus. Dla latest również dominują niedozwolone kontakty. To nie jest dowód, że należy poluzować collider: warunki sukcesu pozostają takie same.</p>
<p>Zarejestrowano nazwy obu geomów, czas i penetrację pierwszego niedozwolonego kontaktu oraz przebiegi wybranych prób. Pomiar kończy się z epizodem, więc szczyty sił porażek mogą być ucięte. Nie używamy ich do twierdzenia, że dane lądowanie jest łagodniejsze lub mniej uszkadza sprzęt.</p>
{videos(video_entries)}
<h3>6. Czy aktor rozróżnia prędkość jeszcze w locie?</h3>
<p>Przy h=2,5 m, tym samym seedzie, zerowym przechyle i prędkościach 0 oraz ±1,5 m/s wzdłuż osi początkowe obserwacje aktora różnią się maksymalnie o około 0,000151. Maksymalna różnica początkowej znormalizowanej akcji wynosi około {probe['best']['initial_max_action_difference']:.6f} dla best i {probe['latest']['initial_max_action_difference']:.6f} dla latest. W pierwszych 0,3 s maksymalne różnice akcji narastają odpowiednio do około {probe['best']['max_action_difference_first_0p3s']:.3f} i {probe['latest']['max_action_difference_first_0p3s']:.3f}.</p>
<p>Model ma opory powietrza: density=1,2 i viscosity=0,00002, więc obserwacje nie są ściśle identyczne i mogą zawierać pośrednie sygnały prędkości. Aktor nadal nie dostaje jej jawnie. Ten test pokazuje małe zróżnicowanie początkowej reakcji, lecz nie dowodzi braku możliwości estymacji ani tego, że dodanie v_xy rozwiąże zadanie. Wymaga to osobnej ablacji treningowej z uczciwie opisanym źródłem pomiaru.</p>
{comparison}
{joint_ranges}
<h3>Wniosek do dalszej pracy</h3>
<p>Zachować oba checkpointy: best jako kandydata lepszego na nowym losowym zestawie łączonym, latest jako punkt odniesienia z lepszą częścią czysto prędkościową. Głównym rezultatem jest teraz zmierzony spadek skuteczności wraz z prędkością i jej kierunkiem oraz identyfikacja ograniczonego obszaru działania. Następne uczenie powinno testować konkretną hipotezę dotyczącą informacji o prędkości lub obsługi touchdown, zamiast powtarzać identyczny profil tylko dłużej.</p>
<p class="caption">Wszystkie wyniki dotyczą jednego seeda treningu, tego samego modelu miękkiego kontaktu i określonych rozkładów stanów. Nie są walidacją sprzętową. Dwa filmy wybrano jako ilustracje po analizie pełnego zestawu; nie służą do szacowania skuteczności.</p>
<p><a href="../artifacts/v3_checkpoint_experiments_2026-10-26/summary.json">Podsumowanie i przedziały</a> · <a href="../artifacts/v3_checkpoint_experiments_2026-10-26/results.json">Pełne próby główne i kontakty</a> · <a href="../artifacts/v3_checkpoint_experiments_2026-10-26/protocol.json">Protokół i hashe kodu</a> · <a href="../artifacts/v3_checkpoint_experiments_2026-10-26/region_protocol.json">Protokół ograniczonej domeny</a> · <a href="../artifacts/v3_checkpoint_experiments_2026-10-26/observability_probe.json">Próba obserwowalności</a></p>
</section>'''
