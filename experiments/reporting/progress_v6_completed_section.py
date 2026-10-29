"""Completed v6 run and independently paired diagnostic experiments."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v6_completed_9398616_2026-10-29'
BASE='../artifacts/v6_completed_9398616_2026-10-29/'
def build(table,videos):
 s=json.loads((OUT/'test_summary.json').read_text());p=json.loads((OUT/'parsed.json').read_text());m=json.loads((OUT/'checkpoints.json').read_text());last=p['rows'][-1]['metrics'];movies=json.loads((OUT/'videos.json').read_text())
 def stat(n,k):return next(r for r in s['groups'] if r['controller']==n and r['kind']==k)
 def score(r):return f'{r["successes"]}/{r["count"]} · {100*r["rate"]:.1f}%'
 cluster=[]
 for n in ['best','latest']:
  v=m[n]['metrics'];cluster.append([n,f'{v["nr_env_steps"]:,}'.replace(',',' ')]+[f'{100*v[f"eval/{k}/policy/success_rate"]:.1f}%' for k in ['nominal','angles','velocity','perturbed']])
 groups=[[label]+[score(stat(n,k)) for n in ['zero','best','latest']] for k,label in [('nominal','Nominalne wysokości 1–3 m'),('random_angles','3 m, same kąty ±30°'),('random_velocity','3 m, sama prędkość w kole ≤5 m/s'),('random_combined','3 m, oba czynniki'),('combined_lower','2,2 m, te same kąty i prędkości co wyżej')]]
 speed=[];angles=[]
 for key,values,out in [('speed',[1,2,3,4,5],speed),('angle',[10,20,30],angles)]:
  for v in values:
   row=[str(v)+(' m/s' if key=='speed' else '°')]
   for n in ['zero','best','latest']:
    r=next(r for r in s[key+'_grid'] if r['controller']==n and r[key]==v);row.append(f'{r["successes"]}/{r["count"]}')
   out.append(row)
 pr=s['previous_domain'];films=[dict(title='V6: PD/zero i best na 3 m',src=movies[0]['src'],note='Nominalnie, roll=30°, vy=3 m/s oraz pierwszy losowy przypadek połączony. Bez selekcji po wyniku.'),dict(title='V6: best i latest na tych samych stanach',src=movies[1]['src'],note='Te same cztery przypadki. Obraz panelu zatrzymuje się po zakończeniu epizodu; film 4× wolniej.')]
 return f'''<section id="e13" class="stage"><div class="stage-label">Etap 13 / 29.10.2026 / v6 zakończony · zadanie 9398616</div>
<h2>V6 opanował nominalne 3 m, ale nie pełne zaburzenia.</h2>
<p class="lead">Trening zakończył 30 015 488 przejść. Walidacja na 3 m: best 15,6%, latest 14,1% przy jednoczesnych roll/pitch ±30° i module prędkości do 5 m/s. Nominalne 3 m zaliczają oba modele. Niski wynik dotyczy rozszerzonej domeny, a nie bezpośrednio tego samego testu, na którym v5 uzyskał 96,1%.</p>
<h3>Rzeczywiste konfiguracje i zakończenie treningu</h3>
<p>Potwierdzono idealny pomiar heading_xy: 80 wejść aktora, 94 critic, skala /4 m/s, szum i opóźnienie zero. Wysokość bazy 1–3 m; prędkość losowana po polu koła do 5 m/s; roll i pitch niezależnie do ±30°. Zachowano PD606, fizykę implicitfast 2,5 ms, 50 Hz i kryteria bezpieczeństwa. Best i latest mają różne przywrócone tablice wag.</p>
<p>Log zawiera {len(p['rows'])} agregatów, bez Traceback, CUDA OOM i wpisów NaN. Zapisany czas: {p['first_log_time']}–{p['last_metrics_time']}. Skrócenie czasu względem v5 nie oznacza automatycznie lepszego uczenia: nieudane próby kończą się szybciej. Best pochodzi z 26 214 400 przejść, latest z końca budżetu.</p>
{table(['Model','Przejścia','Nominalnie','Same kąty','Sama prędkość','Oba czynniki'],cluster)}
<p>To zestaw walidacyjny używany do wyboru best, po 64 próby na warunek/politykę. Best wygrał w połączonych zaburzeniach, choć latest poprawił wyniki osobnych czynników. Różnica 10/64 i 9/64 nie dowodzi trwałej przewagi jednego checkpointu.</p>
<h3>Dlaczego większa domena okazała się trudna?</h3>
<p>Jednocześnie zwiększono wysokość do 3 m, kąty z ±10° do ±30° na oś oraz prędkość i udział dużych modułów. W kole o promieniu 5 m/s aż 64% masy rozkładu przypada powyżej 3 m/s. Przy obu kątach skrajnych całkowity przechył przekracza 30°. Nie jest to izolowana zmiana jednego parametru ani porównanie z v5 na równym zadaniu.</p>
<p>Końcowa średnia trudność w 64 środowiskach wynosi dla kątów {last['env_info/curriculum/angle_difficulty']:.3f}, a dla prędkości {last['env_info/curriculum/velocity_difficulty']:.3f}. Pełny zakres w walidacji jest stały, podczas gdy curriculum w treningu nadal dobiera łatwiejsze warunki. Nie oznacza to błędu ewaluacji. Agregaty trudności są średnimi, a pola amplitud uwzględniają też grupy z zerowym zaburzeniem — nie są jednym globalnym limitem wszystkich epizodów.</p>
<p>Nie widać powrotu dawnego naprzemiennego nasycenia łydek: oba wskaźniki przedkontaktowe są zerowe w agregatach. Końcowe std={last['policy/std_dev']:.4f}, KL={last['policy_ratio/approx_kl']:.4f}, clip fraction={100*last['policy_ratio/clip_fraction']:.1f}%, explained variance={last['v_value/explained_variance']:.3f}. Sam wysoki explained variance nie oznacza skutecznej polityki; większe KL i clip fraction pokazują aktywne aktualizacje, ale nie dowodzą przyczyny słabego wyniku.</p>
<figure class="chart"><img src="{BASE}analysis.svg" alt="Uczenie v6, curriculum i osobne testy kątów i prędkości"><figcaption>Góra: walidacja i średnie curriculum. Dół: osobne deterministyczne siatki diagnostyczne na 3 m.</figcaption></figure>
<h3>Niezależny od walidacji zestaw diagnostyczny</h3>
<p>Wykonano 314 scenariuszy na metodę: 942 epizody dla PD/zero, best i latest. Nowy seed scenariuszy, zamrożona fizyka, idealna obserwacja prędkości, deterministyczne akcje. W każdej trójce potwierdzono identyczny hash scenariusza. W parze 3 m/2,2 m zachowano te same kąty i prędkości; wysokość jest zmienianym czynnikiem. Przy licznościach po 64 próby nie należy interpretować różnic kilku sukcesów jako stabilnej przewagi.</p>
{table(['Warunki','PD / zero','Best v6','Latest v6'],groups)}
<p>Dla połączonych przypadków na 3 m przedziały Wilsona 95%: best {100*stat('best','random_combined')['ci95'][0]:.1f}–{100*stat('best','random_combined')['ci95'][1]:.1f}%, latest {100*stat('latest','random_combined')['ci95'][0]:.1f}–{100*stat('latest','random_combined')['ci95'][1]:.1f}%. Dotyczą próbkowania scenariuszy, nie zmienności treningu między seedami.</p>
{table(['Moduł prędkości przy zerowych kątach','PD / zero','Best v6','Latest v6'],speed)}
{table(['Pojedyncza oś: oba znaki roll/pitch','PD / zero','Best v6','Latest v6'],angles)}
<p>Rozkład przyczyn wszystkich porażek: PD {s['failures']['zero']}; best {s['failures']['best']}; latest {s['failures']['latest']}. Obowiązują niezmienione ograniczenia kontaktu, sił i przyspieszenia oraz okno stabilności. Nie obniżano kryteriów, aby uzyskać wyższy procent.</p>
<p>Na tych samych 64 połączonych stanach latest poprawia wynik z 12/64 (18,8%) na 3 m do 32/64 (50,0%) na 2,2 m. W tym podzbiorze wysokość jest istotnym czynnikiem trudności. Osobna siatka prędkości pokazuje 8/8 przy 1–2 m/s, 5/8 przy 3 m/s, 1/8 przy 4 m/s i 0/8 przy 5 m/s dla latest. Przy pojedynczym kącie 30° oba modele zaliczają tylko 1/4 przypadków. To lokalne siatki diagnostyczne, a nie ciągłe gwarancje granic skuteczności.</p>
<h3>Czy utracono wcześniejsze umiejętności?</h3>
<p>Na stałych pierwszych 16 połączonych przypadkach z każdej z czterech wysokości starego benchmarku (64 próby, zakres v5: ±10°, vx/vy ±3 m/s) archiwalny v5 latest ma {pr['v5_latest']}/64, v6 best {pr['best']}/64, a v6 latest {pr['latest']}/64. Potwierdzono identyczne hashe resetów. To retrospektywny podzbiór diagnostyczny, nie nowy pełny test końcowy; pozwala oddzielić zwiększenie wymagań od ewentualnej utraty części wcześniejszej skuteczności.</p>
<h3>Filmy bez wyboru po wyniku</h3>
{videos(films)}
<p>Sprawdzono zgodność {s['video_outcomes_verified']} rozstrzygnięć, czasów i hashy nagrań z tabelą eksperymentów.</p>
<h3>Decyzja: przygotować v7 niezależnie od wyniku v6</h3>
<p>Zgodnie z decyzją użytkownika następny trening zachowuje rozszerzoną domenę i dodaje szum oraz lekkie opóźnienie prędkości. Nie przedstawiamy tego jako naprawy niskiej skuteczności v6. To kolejny eksperyment nad realistycznością dostępnej obserwacji; jego przygotowanie i testy są w etapie 14.</p>
<p><a href="{BASE}test_summary.json">Podsumowanie</a> · <a href="{BASE}paired_results.csv">Sparowane wyniki CSV</a> · <a href="{BASE}best_protocol.json">Protokół 314 scenariuszy</a> · <a href="{BASE}parsed.json">Przebieg treningu</a> · <a href="{BASE}checkpoints.json">Checkpointy</a> · <a href="{BASE}previous_domain_probe.json">Kontrola dawnej domeny</a> · <a href="{BASE}videos.json">Metadane filmów</a></p></section>'''
