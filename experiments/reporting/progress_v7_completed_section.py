"""V7 completion with matched physical scenarios and sensor interventions."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v7_completed_9400825_2026-10-29'
BASE='../artifacts/v7_completed_9400825_2026-10-29/'
def build(table,videos):
 s=json.loads((OUT/'test_summary.json').read_text());p=json.loads((OUT/'parsed.json').read_text());meta=json.loads((OUT/'checkpoints.json').read_text());last=p['rows'][-1]['metrics'];movie=json.loads((OUT/'videos.json').read_text())[0]
 def stat(method,kind):return next(r for r in s['groups'] if r['method']==method and r['kind']==kind)
 def score(r):return f'{r["successes"]}/{r["count"]} · {100*r["rate"]:.1f}%'
 groups=[[label]+[score(stat(n,k)) for n in ['v6_best','v6_latest','v7_noisy']] for k,label in [('nominal','Nominalne wysokości 1–3 m'),('random_angles','3 m, kąty ±30°, bez prędkości'),('random_velocity','3 m, moduł prędkości ≤5 m/s, bez kątów'),('random_combined','3 m, kąty i prędkość jednocześnie'),('combined_lower','2,2 m, te same kąty i prędkości')]]
 matrix=[]
 for sensor,label in [('clean','Idealny pomiar'),('noise_only','Tylko szum σ=0,1 m/s'),('delay_only','Tylko opóźnienie 20 ms'),('noisy','Szum + opóźnienie')]:
  row=[label]
  for training in ['v6','v7']:
   checkpoint='best' if training=='v7' and sensor in ['clean','noisy'] else 'latest';r=next(r for r in s['sensor_matrix'] if r['training']==training and r['sensor']==sensor and r['checkpoint']==checkpoint);row.append(score(r))
  matrix.append(row)
 speed=[];angles=[]
 for key,values,out in [('speed',[1,2,3,4,5],speed),('angle',[10,20,30],angles)]:
  for v in values:
   row=[f'{v:g}'+(' m/s' if key=='speed' else '°')]
   for name in ['v6_latest','v7_noisy']:
    r=next(r for r in s[key+'_grid'] if r['method']==name and r['value']==v);row.append(f'{r["successes"]}/{r["count"]}')
   out.append(row)
 pair=next(r for r in s['paired_sensor_comparisons'] if r['comparison']=='v6/v7 both noisy');v7pair=next(r for r in s['paired_sensor_comparisons'] if r['comparison']=='v7 clean/noisy');full=stat('v7_noisy','random_combined')
 return f'''<section id="e15" class="stage"><div class="stage-label">Etap 15 / 29.10.2026 / zakończony v7 · zadanie 9400825</div>
<h2>V7 toleruje model pomiaru, ale pełna domena nadal jest trudna.</h2>
<p class="lead">Trening zakończył 30 015 488 przejść. Walidacja na 3 m: 18,75% przy obu zaburzeniach, 68,75% przy samych kątach i 43,75% przy samej prędkości; nominalnie 100%. Osobny test daje 14/64 (21,9%) sukcesów połączonych. Best i latest mają identyczne przywrócone wagi polityki.</p>
<h3>Co dokładnie potwierdzono w pobranych plikach?</h3>
<p>Best i latest pochodzą z końcowych 30 015 488 przejść. Różnią się plikowym SHA256, ale hash tablic parametrów polityki jest identyczny. Wykonano jeden pełny benchmark oraz {s['latest_replays']} dodatkowych odtworzeń latest, porównując wyniki, czasy i metryki. Nie liczymy identycznych wag jako dwóch niezależnych modeli.</p>
<p>Konfiguracja zadania i PPO jest identyczna z v6: wysokość 1–3 m, niezależne roll/pitch do ±30°, prędkość losowana po polu koła do 5 m/s, PD606, implicitfast 2,5 ms i sterowanie 50 Hz. Zmienił się wyłącznie model pomiaru dwóch prędkości aktora: szum σ=0,1 m/s na składową, brak biasu i jeden krok opóźnienia (20 ms). Critic zachowuje uprzywilejowane obserwacje. Oba treningi mają seed=1; nie są powtórzeniami po seedach.</p>
<p>Log ma {len(p['rows'])} agregatów, bez Traceback, CUDA OOM i wpisów NaN. Zapisane czasy: {p['first_log_time']}–{p['last_metrics_time']}. Oba przedkontaktowe wskaźniki oscylacji łydek pozostają zerowe. Końcowa średnia trudność kątów {last['env_info/curriculum/angle_difficulty']:.3f}, prędkości {last['env_info/curriculum/velocity_difficulty']:.3f}; curriculum nadal nie dotarło średnio do pełnej domeny.</p>
<p>Końcowe std={last['policy/std_dev']:.4f}, KL={last['policy_ratio/approx_kl']:.4f}, clip fraction={100*last['policy_ratio/clip_fraction']:.1f}%, explained variance={last['v_value/explained_variance']:.3f}. Są zbliżone do v6. Średni zapisany błąd prędkości na końcu wynosi {last['env_info/observation/velocity_error_norm']:.3f} m/s; zawiera szum i zmianę prędkości w czasie opóźnienia, nie jest samym σ sensora.</p>
<figure class="chart"><img src="{BASE}analysis.svg" alt="V6 i v7: walidacja, curriculum, test prędkości i kątów"><figcaption>Góra: walidacja w macierzystym modelu pomiaru każdego treningu. Dół: osobne siatki diagnostyczne na tych samych fizycznych stanach. V6 ma tu idealny pomiar, v7 szum i opóźnienie — kontrolę wspólnego sensora pokazuje dalsza tabela.</figcaption></figure>
<h3>Ten sam benchmark co dla v6: 314 scenariuszy</h3>
<p>V7 wykonał 314 nowych epizodów na dokładnie tych samych stanach początkowych i przy tej samej fizyce co poprzedni test v6. W każdej parze sprawdzono hash scenariusza. Powtórnie wykorzystano archiwalne wyniki v6; v7 działa z własnym modelem szumu i opóźnienia. Jest to znany benchmark projektu, nie nowy nigdy wcześniej oglądany test końcowy. Dla v7 występuje jedna deterministycznie odtwarzalna realizacja szumu na scenariusz.</p>
{table(['Warunki','V6 best / ideal','V6 latest / ideal','V7 best=latest / noisy'],groups)}
<p>Wynik v7 przy obu czynnikach na 3 m ma przedział Wilsona 95% {100*full['ci95'][0]:.1f}–{100*full['ci95'][1]:.1f}%. Opisuje próbkę scenariuszy, nie zmienność między seedami uczenia ani osobno wariancję realizacji szumu. Mała przewaga nad v6 nie uzasadnia tezy, że dodanie szumu poprawia uczenie.</p>
{table(['Moduł prędkości; osiem kierunków, bez kątów','V6 latest / ideal','V7 / noisy'],speed)}
{table(['Wartość kąta; obie osie i oba znaki','V6 latest / ideal','V7 / noisy'],angles)}
<h3>Kontrolowana interwencja: ten sam sensor dla obu polityk</h3>
<p>Na 64 tych samych połączonych przypadkach na 3 m porównano idealny pomiar, sam szum, samo opóźnienie oraz oba efekty. Nie zmieniano wag. Modele wczytano w zgodnych środowiskach macierzystych, a następnie jawnie podano obserwacje z alternatywnego sensora o tej samej ramie, skali i układzie wejść. Nie osłabiano zabezpieczenia zgodności checkpointów i nie nazywamy tych testów ponownym treningiem.</p>
{table(['Pomiar przy inferencji','V6 latest','V7 best=latest'],matrix)}
<figure class="chart"><img src="{BASE}sensor_matrix.svg" alt="Rozdzielenie samego szumu i samego opóźnienia dla v6 i v7"><figcaption>Te same 64 fizyczne przypadki w każdej kolumnie; wartości podane też jako liczby sukcesów. To niewielki test diagnostyczny, nie dowód statystycznej równoważności sensorów.</figcaption></figure>
<p>Przy pełnym modelu noisy v6 latest uzyskuje 13/64, v7 14/64. W parach: {pair['both_success']} wspólnych sukcesów, {pair['a_only']} tylko v6, {pair['b_only']} tylko v7, {pair['both_fail']} wspólnych porażek. V7 z idealną prędkością także ma 14/64; w tej próbce wszystkie rozstrzygnięcia są identyczne: {v7pair['both_success']} wspólnych sukcesów i {v7pair['both_fail']} wspólnych porażek. Identyczne rozstrzygnięcia nie oznaczają identycznych trajektorii i obciążeń ani niewrażliwości na dowolny model pomiaru.</p>
<p>W tej próbce usunięcie szumu i opóźnienia nie zbliża v7 do 70–80%. Główną przeszkodą pozostaje opanowanie manewru w pełnej domenie 3 m / ±30° / 5 m/s. Nie dowodzi to, że dowolny błąd lub większe opóźnienie są bezpieczne; sprawdzono tylko podane wartości. Wykonano {s['intervention_episodes']} dodatkowych epizodów interwencji sensora.</p>
<h3>Charakter porażek i interpretacja dla artykułu</h3>
<p>W 314 próbach v7: {s['failures']['v7_noisy']}. Obowiązują dotychczasowe limity kontaktów, przyspieszeń i stabilności. Nominalne 3 m pozostaje osiągalne, ale połączenie dużego przechyłu, poziomego pędu i energii zrzutu wymaga więcej niż dokładniejszej obserwacji. W v6 50% połączonych prób przy 2,2 m spadało do 18,8% przy 3 m; w v7 ten efekt oceniamy na tych samych sparowanych wysokościach w tabeli.</p>
<p>Wniosek naukowy: dla przyjętych σ i 20 ms obserwacja prędkości pozostaje użyteczna, ale ten przebieg nie potwierdza skutecznego landingu w całej rozszerzonej domenie. Warunki trzeba raportować oddzielnie, zamiast zestawiać 21,9% v7 z 96,1% v5 z łatwiejszego testu. Do mocnego porównania v6/v7 potrzebne są kolejne seedy treningu i nowe realizacje szumu.</p>
<h3>Film: jednakowy noisy sensor w obu panelach</h3>
{videos([dict(title='V6 latest i v7: ten sam model pomiaru na 3 m',src=movie['src'],note='Szum σ=0,1 m/s i opóźnienie 20 ms w obu panelach. Cztery stałe przypadki: nominalny, roll=30°, boczne 3 m/s i pierwszy losowy połączony. Zachowano również porażki; film 4× wolniej.')])}
<p>Osiem rozstrzygnięć filmu odtworzono dodatkowo i sprawdzono z czasami i hashami resetów.</p>
<h3>Co dalej?</h3>
<p>Najpierw warto rozdzielić postęp wysokości od kątów i prędkości oraz sprawdzić, czy polityka osiąga pełne curriculum. Dotychczasowe krzywe nadal rosną, lecz sam dodatkowy budżet nie gwarantuje rozwiązania. Zachowanie bezpiecznej domeny i osobne testy graniczne pozwolą ocenić, czy barierą jest ekspozycja na trudne przypadki, strategia ruchu czy fizyczne ograniczenia napędów i kontaktu. Parametry rzeczywistego OptiTrack nadal wymagają pomiaru; v7 testował założony model sensora. Nie przygotowano automatycznie kolejnego treningu.</p>
<p><a href="{BASE}test_summary.json">Podsumowanie</a> · <a href="{BASE}paired_results.csv">Wspólne scenariusze CSV</a> · <a href="{BASE}native_results.json">314 epizodów v7</a> · <a href="{BASE}sensor_components_results.json">Szum i opóźnienie osobno</a> · <a href="{BASE}v6_noisy_results.json">V6 z noisy</a> · <a href="{BASE}v7_clean_results.json">V7 z idealnym pomiarem</a> · <a href="{BASE}configuration_comparison.json">Zgodność konfiguracji</a> · <a href="{BASE}checkpoints.json">Checkpointy i hashe wag</a> · <a href="{BASE}parsed.json">Przebieg treningu</a> · <a href="{BASE}latest_verification.json">Odtworzenie latest</a> · <a href="{BASE}videos.json">Metadane filmu</a></p></section>'''

def overview():
 return '''<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan / zakończony v7 · 9400825</div><h2>Szum i 20 ms opóźnienia nie są główną barierą w zbadanej próbce.</h2><p class="lead">V7 zakończył 30 mln przejść. Best i latest mają identyczne wagi. Walidacja połączonych warunków na 3 m: 18,75%; osobny benchmark: 14/64 (21,9%). Przy jednakowym noisy sensorze v6 latest ma 13/64. To niewielka różnica, nadal poniżej celu.</p><p>V7 z idealnym pomiarem też uzyskuje 14/64. Osobne interwencje szumu i opóźnienia, wyniki wysokości/kątów/prędkości i film porównawczy są w etapie 15. Wyniki dotyczą jednego seeda uczenia i założonego, nie skalibrowanego modelu OptiTrack.</p><p><a href="#e15">Etap 15: pełna analiza v7 i modelu pomiaru →</a> · <a href="#e13">Zakończony v6</a> · <a href="#e11">Wcześniejszy v5</a></p></section>'''
