"""Complete paired condition tables and article interpretation from measured trials."""
import csv
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v3_checkpoint_experiments_2026-10-26'
NAMES=['zero','best','latest']


def build(table):
    rows=[]
    for filename in ['results.json','refinement_results.json','region_results.json']:
        rows.extend(json.loads((OUT/filename).read_text()))
    cases={}
    for r in rows:cases.setdefault(r['case']['id'],{})[r['controller']]=r
    assert len(cases)==656
    for grouped in cases.values():
        assert set(grouped)==set(NAMES)
        assert len({r['scenario_hash'] for r in grouped.values()})==1
    def result(selected,name):
        selected=[r for r in selected if r['controller']==name]
        k=sum(r['success'] for r in selected);n=len(selected)
        return f'{k}/{n} ({100*k/n:.1f}%)'
    methods=table(['Metoda','Działanie','Co porównujemy'],[
        ['PD / zero akcji','a=0; stała pozycja zadana home','Bazowy regulator przygotowany przez search; silniki aktywne'],
        ['PPO best + PD','a=π_best(o); zmienne pozycje zadane','Wagi z kroku 27 262 976'],
        ['PPO latest + PD','a=π_latest(o); zmienne pozycje zadane','Wagi z kroku 30 015 488']])
    groups=[('nominal','Nominalne','h=1…2,5 m co 0,1 m; kąty=0; v_xy=0'),('angle','Same kąty','h∈{1;1,5;2;2,5} m; osiem kombinacji roll/pitch 0 i ±3°; v_xy=0'),('speed','Sama prędkość','cztery wysokości; |v_xy|=0,5;1;1,5;2;2,5;3 m/s; osiem kierunków; kąty=0'),('refinement','Doprecyzowanie prędkości','cztery wysokości; |v_xy|=1,25 m/s; osiem kierunków; kąty=0'),('heldout','Łączone — pełny zakres','cztery wysokości; roll/pitch U[−3°,3°]; vx i vy niezależnie U[−3,3] m/s'),('limited_region','Łączone — ograniczony zakres','cztery wysokości; roll/pitch U[−3°,3°]; moduł U[0,1] m/s i losowy kierunek')]
    # File-specific kind labels are taken from the saved protocol, not inferred from IDs.
    refinement_ids={r['case']['id'] for r in json.loads((OUT/'refinement_results.json').read_text())}
    region_ids={r['case']['id'] for r in json.loads((OUT/'region_results.json').read_text())}
    def kind(r):
        return 'refinement' if r['case']['id'] in refinement_ids else 'limited_region' if r['case']['id'] in region_ids else r['case']['kind']
    overall=[];by_height=[]
    for key,label,condition in groups:
        selected=[r for r in rows if kind(r)==key]
        overall.append([label,condition]+[result(selected,n) for n in NAMES])
        for h in [1.,1.5,2.,2.5]:
            sub=[r for r in selected if r['case']['height']==h]
            by_height.append([label,f'{h:g} m']+[result(sub,n) for n in NAMES])
    detailed=[];export=[]
    for case_id,grouped in sorted(cases.items()):
        c=grouped['zero']['case'];out=[str(case_id),kind(grouped['zero']),f'{c["height"]:g}',f'{c["roll"]:.3f}',f'{c["pitch"]:.3f}',f'{c["vx"]:.3f}',f'{c["vy"]:.3f}']
        out.extend('Sukces' if grouped[n]['success'] else grouped[n]['reason'] for n in NAMES)
        detailed.append(out)
        item={**c,'test_group':kind(grouped['zero']),'scenario_hash':grouped['zero']['scenario_hash']}
        for n in NAMES:
            item.update({n+'_success':grouped[n]['success'],n+'_reason':grouped[n]['reason'],n+'_end_time':grouped[n]['end_time']})
        export.append(item)
    keys=list(dict.fromkeys(k for r in export for k in r));buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=keys);writer.writeheader();writer.writerows(export)
    (OUT/'paired_conditions.csv').write_text(buf.getvalue())
    (OUT/'paired_conditions.json').write_text(json.dumps(export,indent=2))
    return f'''<h3 id="porownanie-metod">Kompletne porównanie metod i warunków początkowych</h3>
<p>W tej implementacji <strong>samo PD i zerowa akcja są tą samą metodą</strong>. Zerowanie akcji usuwa korektę polityki, ale nie wyłącza regulatora ani momentów silników. Nie liczymy ich jako dwóch niezależnych baseline’ów.</p>
{methods}
<p>Sygnał zadany ma postać q_ref=clip(q_home+s·a), z uwzględnieniem offsetów resetu i przełożenia aktuatora. Dla a=0 regulator utrzymuje home; dla PPO zmienia się pozycja zadana, a nie Kp/Kd. Porównanie jest więc ablacją korekty polityki na wspólnej bazie PD. Nie badano tu swobodnego spadku z wyłączonymi silnikami ani regulatora z nastawami dobieranymi osobno do każdego stanu.</p>
<h4>Wspólne warunki i kompletność zakresu</h4>
<p>Każdy z 656 stanów odtworzono dla wszystkich trzech metod. Hash resetu jest zgodny między metodami. Zmieniano wysokość, roll/pitch oraz vx/vy według tabeli; yaw=0, początkowe vz=0. Nie zadawano dodatkowego zaburzenia prędkości kątowej. Model, grawitacja, miękki kontakt, podział collidera łydki, PD, limity, czas sterowania 20 ms i definicja sukcesu są wspólne. To pełne porównanie wykonanych zestawów, a nie pełna siatka wszystkich kombinacji stanu 6-DoF.</p>
{table(['Zestaw','Warunki początkowe','PD / zero akcji','PPO best + PD','PPO latest + PD'],overall)}
<p>Nie sumujemy skuteczności wszystkich zestawów do jednego wyniku: ich udział wynika z projektu eksperymentu, a nie z rozkładu rzeczywistych zdarzeń awaryjnych. Zestawy doprecyzowania i ograniczonej domeny wybrano po obejrzeniu wyników głównych.</p>
<h4>Każdy zestaw w podziale na wysokości</h4>
{table(['Zestaw','Wysokość','PD / zero akcji','PPO best + PD','PPO latest + PD'],by_height)}
<p>Wiersze nominalne poniżej obejmują tylko cztery wysokości wspólne pozostałym zestawom; pełny test nominalny ma 16 wysokości. Wyniki prędkości zagregowane po wysokości należy czytać razem z tabelą prędkości i mapą kierunków powyżej. Zestaw pełny losuje składowe niezależnie, więc moduł może sięgać √18≈4,24 m/s; ograniczony zestaw losuje moduł do 1 m/s. Tych domen nie można utożsamiać.</p>
<details><summary>Wszystkie 656 sparowanych stanów i wyniki każdej metody</summary>
<p>Kąty w stopniach, wysokość w metrach, prędkości w m/s. Wyświetlane wartości zaokrąglono; eksport zachowuje pełną precyzję i hashe. „Sukces” oznacza spełnienie implementowanego kryterium stabilnego lądowania; inne wpisy podają przyczynę zakończenia.</p>
{table(['ID','Zestaw','h','roll','pitch','vx','vy','PD / zero','Best','Latest'],detailed)}</details>
<p><a href="../artifacts/v3_checkpoint_experiments_2026-10-26/paired_conditions.csv">Pobierz kompletne porównanie CSV</a> · <a href="../artifacts/v3_checkpoint_experiments_2026-10-26/paired_conditions.json">JSON z pełną precyzją i hashami</a></p>
<h3 id="wnioski-artykul">Co te eksperymenty wnoszą do artykułu?</h3>
{table(['Pytanie badawcze','Wniosek poparty danymi','Granica interpretacji'],[
['RQ1 — zakres działania','PPO zachowuje nominalne lądowanie 1–2,5 m. W eksploracyjnej domenie |v_xy|≤1 m/s i ±3° uzyskano 125/128 oraz 127/128 sukcesów.','Pełny zakres vx/vy ±3 m/s daje 63/256 i 55/256; cel 70–80% nie jest spełniony. Mała domena wymaga zamrożonego testu potwierdzającego.'],
['RQ2 — wysokość, orientacja, prędkość','Same przechyły ±3° są obsługiwane w 32/32 próbach obu polityk. Skuteczność silnie zależy od prędkości i kierunku; spadek pojawia się około badanych punktów 1,25–2 m/s.','Nie jest to ciągła granica ani dowód dla większych przechyłów. Wpływ wysokości zależy od zestawu; brak podstaw do prostego prawa monotonicznego.'],
['RQ3 — faktoryzacja curriculum','Przetestowano politykę wytrenowaną z rozdzielonymi grupami zaburzeń i zmierzono jej zakres działania.','Sam ten eksperyment nie dowodzi przewagi faktoryzacji: potrzeba kontrolowanej ablacji curriculum i wielu seedów treningu. Best/latest to jeden trening.'],
['RQ4 — informacja sensoryczna','Próba obserwowalności pokazuje bardzo zbliżoną początkową reakcję mimo różnych prędkości XY.','Brak jawnego v_xy jest hipotezą ograniczenia. Potrzebna ablacja obserwacji; opory powietrza dostarczają sygnałów pośrednich.'],
['Jakość kontaktu','W porażkach dominują niedozwolone kontakty, często obudów przednich ud.','Nie wykazano mniejszych uszkodzeń ani bezpieczeństwa sprzętowego. Epizody porażek kończą się wcześniej, co ucina pomiary obciążeń.']])}
<p><strong>Proponowany wniosek do sekcji Results/Discussion:</strong> polityka PPO, sterująca korektą pozycji zadanych nad przygotowanym regulatorem PD, nauczyła się stabilnego lądowania w symulacji z wysokości 1–2,5 m w zbadanych warunkach nominalnych i wykazuje wysoką skuteczność w ograniczonym zakresie małych przechyłów i prędkości. Eksperymenty rozdzielające czynniki wskazują prędkość poziomą i jej kierunek jako główne ograniczenie badanego zakresu. Przy łączonych zaburzeniach pełnej domeny skuteczność pozostaje niska. Wkładem jest charakterystyka zakresu działania polityki emergency landingu; PD stanowi bazę sterowania i ablacją pokazujemy rolę korekty polityki, bez formułowania celu jako wyższości PPO nad search.</p>
<p>Do końcowego artykułu należy zamrozić domenę oraz próg sukcesu przed testem, wybrać checkpoint na walidacji, przeprowadzić niezależne testy i powtórzenia treningu. Osobno potrzebne są kontrolowane ablacje curriculum i informacji o prędkości oraz analiza obciążeń z uwzględnieniem uciętych trajektorii. Obecne dane nie obejmują dużych obrotów, zadanych prędkości kątowych, niezerowego początkowego vz, nierównego podłoża ani walidacji sprzętowej.</p>'''
