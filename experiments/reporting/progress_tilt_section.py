"""Report the measured tilt diagnostic and timestep replay, without causal overclaiming."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/tilt_control_diagnostic'
BASE='../artifacts/tilt_control_diagnostic/'

def build(table):
    s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'timestep_probe.json').read_text());names=['zero','best','latest']
    scores=[];tracking=[]
    for a in [0,3,10,20,30]:
        group=[next(r for r in s['groups'] if r['controller']==n and r['angle']==a) for n in names]
        scores.append([f'{a}°']+[f'{r["successes"]}/{r["count"]} ({100*r["successes"]/r["count"]:.1f}%)' for r in group])
        tracking.append([f'{a}°']+[f'{r["median_flight_tracking_rms_deg"]:.2f}°' for r in group])
    replay=[]
    for r in p['rows']:
        replay.append([r['controller'],r['axis'],f'{1000*r["dt"]:g} ms',f'{100*r["FL_calf_saturated_fraction"]:.1f}%',f'{100*r["FL_calf_sign_flip_fraction"]:.1f}%'])
    detailed=[]
    for r in s['results']:
        c=r['case'];m=r['metrics']['flight'];nc=r['nominal_comparison'];t=r['first_contact']['time'] if r['first_contact'] else None
        detailed.append([r['controller'],f'{c["height"]:g}',c['axis'],f'{c["angle"]:+g}°','sukces' if r['success'] else r['reason'],f'{t:.3f}' if t is not None else 'brak',f'{r["rpy_before_contact"][0]:+.1f}° / {r["rpy_before_contact"][1]:+.1f}°',f'{m["tracking_rms_deg"]:.2f}°',f'{100*m["motor_saturation_any_fraction"]:.1f}%',f'{nc["peak_target_difference_flight_deg"]:.1f}°'])
    figures=''.join(f'<details><summary>Przebiegi: h={x["case"]["height"]:g} m, {x["case"]["axis"]}={x["case"]["angle"]:+}°</summary><figure class="chart"><img loading="lazy" src="{BASE+x["figure"]}" alt="Orientacja, akcje, śledzenie pozycji i momenty w czasie"><figcaption>Linie pionowe oznaczają pierwszy kontakt każdej metody. Dolne wykresy pokazują staw o największym błędzie RMS w locie dla danej polityki. Wybór stawu jest diagnostyczny, na podstawie wyników; wybór ośmiu warunków był stały: wysokości 1 i 2,5 m, roll/pitch +10 i +30°.</figcaption></figure></details>' for x in s['examples'])
    return f'''<section id="e6" class="stage"><div class="stage-label">Etap 06 / test przechyłów i dynamiki PD</div>
<h2>Polityka reaguje w locie. W napędach łydki wykryto oscylacje zależne od kroku fizyki.</h2>
<p class="lead">204 epizody na 68 wspólnych stanach początkowych pokazują dobrą generalizację do części większych przechyłów, ale załamanie skuteczności przy 30°. Rejestracja co 5 ms ujawnia naprzemienne momenty nasycenia w łydkach. Kontrolny replay tych samych pozycji zadanych przy mniejszym kroku usuwa to nasycenie w badanym fragmencie lotu.</p>
<h3>Co sprawdziliśmy?</h3>
<p>PD / zero akcji, best i latest otrzymały identyczne stany: h∈{{1;1,5;2;2,5}} m, roll albo pitch ±3°, ±10°, ±20°, ±30° oraz nominalny start przy każdej wysokości. Drugi kąt, yaw i prędkości liniowe na starcie są zerowe; nie dodawano zaburzenia prędkości kątowej. Każda komórka to jedna deterministyczna próba, a nie estymata skuteczności po wielu seedach. 10–30° leży poza zakresem ±3° treningu tych modeli.</p>
<p>Log zawiera pozycje zadane i rzeczywiste, prędkości stawów, momenty silników, momenty po przełożeniu, roll/pitch/yaw, akcje oraz pierwszy kontakt. Próbkowanie fizyki: 5 ms; sterowanie: 20 ms. Pierwszy kontakt oznacza parę robot–podłoże/podpora z odległością ≤0; osobno zapisano pierwszy kontakt przenoszący ≥1 N. Bufory sił i kontaktu dotyczą kroku całkowania, z niepewnością synchronizacji do 5 ms. Rejestrator nie zmienia fizyki: kontrolny epizod dał identyczny wynik, czas i hash resetu.</p>
<h3>Skuteczność większych przechyłów</h3>
{table(['Moduł przechyłu','PD / zero','PPO best','PPO latest'],scores,'Dla niezerowego kąta: cztery wysokości × dwie osie × dwa znaki = 16 prób na metodę')}
<figure class="chart"><img src="{BASE}success_map.svg" alt="Mapa sukcesów dla roll i pitch w zakresie od minus 30 do plus 30 stopni"><figcaption>Każde pole to pojedynczy scenariusz. Kolumna 0° jest tą samą próbą nominalną na obu panelach i nie jest liczona podwójnie w tabeli.</figcaption></figure>
<p>Oba modele przechodzą 14/16 prób przy 10°, mimo że trening kończył zakres kątów na ±3°. Przy 20° best ma 10/16, latest 7/16; przy 30° odpowiednio 1/16 i 3/16. Nie ma więc podstaw, by uznać brak jakiegokolwiek większego ruchu za główną przyczynę. Wysokość, oś i znak mają znaczenie; mapa pokazuje te różnice bez uśredniania.</p>
<h3>Czy reakcja zaczyna się za późno?</h3>
<p>Dla wszystkich prób ±30° oba modele już od pierwszej decyzji zadają co najmniej jednemu stawowi pozycję różniącą się o ≥5° od własnego przebiegu nominalnego na tej samej wysokości; różnica utrzymuje się przez pierwsze trzy próbki sterowania. Jest to opis reakcji, nie pomiar opóźnienia sensorycznego: nominalne porównanie używa innego seeda resetu, a początkowy przechył jest już obecny w obserwacji.</p>
<p>Mediana największej różnicy pozycji zadanej w locie względem nominalnego przebiegu dla 30° wynosi 22,7° (best) i 34,4° (latest). Reakcja jest wyraźna, ale nie oznacza poprawnej reorientacji. Przykładowo przy h=2,5 m i roll=+30° tuż przed kontaktem roll wynosi około +33,4° dla best i +34,8° dla latest, podczas gdy PD ma +30,5°. Polityki zmieniają konfigurację nóg, lecz w tej próbie nie wyrównują korpusu.</p>
<h3>Czy rzeczywiste stawy nadążają za celem?</h3>
{table(['Moduł przechyłu','PD: RMS w locie','Best: RMS w locie','Latest: RMS w locie'],tracking,'Mediana po epizodach; RMS po wszystkich 13 stawach i próbkach przed kontaktem')}
<p>RMS około 3–4,4° nie jest dowodem, że napędy mają pełen zapas. Uśrednienie po 13 stawach ukrywa lokalne błędy i szybkie zmiany momentu. Dla roll=+30°, h=2,5 m przednia lewa łydka ma RMS około 7,3° (best) i 8,0° (latest); przez większość lotu przynajmniej jeden napęd osiąga ≥95% swojego limitu. W tej próbie nasycenie dotyczy głównie stawów łydki, nie wszystkich silników jednocześnie.</p>
<h3>Kluczowe odkrycie: naprzemienne nasycenie momentów</h3>
<p>Przy obecnym integratorze Euler i kroku 5 ms moment przedniej lewej łydki w analizowanym fragmencie lotu zmienia znak co krok: około −3, +3, −3, +3 Nm po stronie silnika. Jest to cykl około 100 Hz. Występuje również podczas nominalnego lotu PPO, więc sam przechył 30° go nie wywołuje. Te próby używają deterministycznej średniej polityki: oscylacji nie powoduje losowanie std podczas tego testu.</p>
<p>Aby oddzielić decyzje polityki od dynamiki, odtworzono zapisane pozycje zadane przez pierwsze 0,3 s lotu z 2,5 m, dla best/latest oraz startu nominalnego i roll=+30°. Sterowanie nadal zmienia cel co 20 ms; zmieniono wyłącznie krok całkowania na 5, 2,5 lub 1,25 ms, zachowując PD i model. Replay 5 ms odtworzył oryginalne qpos dokładnie (maksymalna różnica 0). Przy drobniejszych krokach zniknęło nasycenie czterech napędów łydki w oknie 0,05–0,30 s.</p>
{table(['Komendy z','Start','Krok fizyki','FL calf: próbki ≥95% limitu','Zmiana znaku między próbkami'],replay,'12 krótkich replayów w locie; nie są to dodatkowe epizody oceny sukcesu. Okno metryk: t>0,05 s do 0,30 s.')}
<figure class="chart"><img src="{BASE}timestep_probe.svg" alt="Ten sam przebieg zadany przy trzech krokach fizyki: moment i pozycja łydki"><figcaption>Przykład latest, roll=+30°, h=2,5 m, FL_calf. Komendy są identyczne i pochodzą z zapisu oryginalnej polityki; podczas replayu polityka nie zamyka pętli.</figcaption></figure>
<p><strong>Interpretacja:</strong> to silna przesłanka wrażliwości numerycznej układu PD–integrator, a nie dowód, że robot fizycznie potrzebuje większych silników. Jednocześnie nie dowodzi, że zmiana dt rozwiąże lądowanie: nie sprawdzono jeszcze zamkniętej pętli ani kontaktu dla drobniejszych kroków. Małe dt w tym podteście służy diagnozie; konfiguracja treningu i podstawowe 204 epizody pozostały przy 5 ms.</p>
<h3>Przebiegi w czasie</h3>{figures}
<details><summary>Wszystkie 204 wyniki i metryki lotu</summary>{table(['Metoda','h [m]','Oś','Kąt','Wynik','Kontakt [s]','Roll / pitch przed kontaktem','RMS śledzenia','Czas z ≥1 napędem przy limicie','Maks. różnica celu od nominalnego'],detailed)}</details>
<h3>Wniosek i następny eksperyment</h3>
<p>Hipoteza „polityka nie wykonuje ruchu przed kontaktem” nie opisuje tych wyników. Dostępny zakres akcji również nie jest jedynym ograniczeniem. Najpierw trzeba rozstrzygnąć stabilność dyskretyzacji PD: wykonać test zbieżności kroku fizyki w zamkniętej pętli oraz podczas kontaktu, utrzymując sterowanie 50 Hz. Dopiero po ustaleniu spójnej dynamiki dla treningu i search warto porównywać większe przechyły i spójną eksplorację. Zmiana dt wymaga zgodności czasowych okien pomiarowych, liczby podkroków, limitowania prędkości, reward i ewaluacji w obu środowiskach.</p>
<p>W artykule obecny test jest diagnostyką generalizacji istniejącej polityki i weryfikacją symulatora. Nie stanowi jeszcze porównania sposobów eksploracji, dowodu fizycznej wykonalności reorientacji ani potwierdzenia poprawy przez mniejsze dt. Wyniki dużych kątów i replayu pochodzą z jednego seeda treningu.</p>
<p><a href="{BASE}protocol.json">Protokół i hashe checkpointów</a> · <a href="{BASE}summary.json">Wyniki i metryki</a> · <a href="{BASE}metrics.csv">Tabela CSV</a> · <a href="{BASE}timestep_probe.json">Kontrola kroku fizyki</a>. Pełne przebiegi NPZ są wskazane przez pole trace w wynikach.</p></section>'''
