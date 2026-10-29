"""Build the project's dark, long-form research report from its preserved data."""
from pathlib import Path
import html
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
REPORT = ROOT / 'docs/report'


def esc(value):
    return html.escape(str(value))


def prose(value):
    text = esc(value)
    links = {
        'docs/article/literature-conditions.md': '../article/literature-conditions.md',
        'docs/article/implementation-agenda.md': '../article/implementation-agenda.md',
        'docs/article/v3-initial-log-9306096.md': '../article/v3-initial-log-9306096.md',
        'docs/experiments/landing_seed606_v3_PLAN.md': '../experiments/landing_seed606_v3_PLAN.md',
    }
    for label,target in links.items():
        text = text.replace(esc(label), f'<a href="{target}">{esc(label)}</a>')
    return text


def paragraphs(items):
    return ''.join(f'<p>{prose(item)}</p>' for item in items)


def bullets(items):
    return '<ul>'+''.join(f'<li>{prose(item)}</li>' for item in items)+'</ul>' if items else ''


def metrics(items):
    return '<dl class="metrics">'+''.join(
        f'<div><dt>{esc(m["label"])}</dt><dd>{esc(m["value"])}</dd></div>' for m in items)+'</dl>'


def table(headers, rows, caption=''):
    return '<div class="table-scroll"><table>'+(
        f'<caption>{esc(caption)}</caption>' if caption else '')+'<thead><tr>'+''.join(
            f'<th scope="col">{esc(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
                '<tr>'+''.join(f'<td>{esc(x)}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def videos(items):
    if not items:
        return ''
    result = '<div class="films">'
    for v in items:
        path=REPORT/v['src']
        assert path.is_file(),path
        poster=path.with_suffix('.png')
        poster_attr=f' poster="{esc(poster.relative_to(REPORT))}"' if poster.exists() else ''
        result+=(f'<figure><video controls playsinline preload="none"{poster_attr} '
                 f'aria-label="{esc(v["title"])}"><source src="{esc(v["src"])}" '
                 f'type="video/mp4">Film: <a href="{esc(v["src"])}">MP4</a></video>'
                 f'<figcaption><strong>{esc(v["title"])}</strong><span>{esc(v.get("note",""))}</span>'
                 f'<a class="download" href="{esc(v["src"])}" download>Pobierz MP4 ↗</a></figcaption></figure>')
    return result+'</div>'


def main():
    data=json.loads((REPORT/'report-data.json').read_text())
    stages=data['stages']
    assert len(stages)>=3
    sections=[]
    stage=stages[0]
    held=json.loads((ROOT/'docs/artifacts/seed606_replay_comparison_2026-10-24/heldout_results.json').read_text())
    comparison=[]
    for h in (1.,1.5,2.,2.5):
        row=[f'{h:g} m']
        for name in ('zero','best','latest'):
            trials=[r for r in held if r['controller']==name and r['height']==h]
            assert len(trials)==32
            count=sum(r['success'] for r in trials)
            row.append(f'{count}/32 · {count/32:.1%}')
        comparison.append(row)
    sections.append(f'''<section id="e1" class="stage"><div class="stage-label">Etap 01 / 24.10.2026</div>
<h2>Od stałego regulatora do polityki PPO</h2>
<p class="lead">PD i search przygotowują bazę uczenia. Badamy, jak polityka PPO wykorzystuje tę bazę podczas awaryjnego lądowania i gdzie kończy się jej skuteczność.</p>
<p>Przed trzecim eksperymentem zakończono dwa treningi. Drugi dodał nominalny replay całego zakresu wysokości. Pobrane best i latest odtworzono na tym samym modelu, z identycznymi Kp/Kd oraz home. Zerowa akcja oznacza utrzymywanie home przez PD; PPO dodaje korekty pozycji stawów.</p>
<h3>Nowe scenariusze z zaburzeniami</h3>
{table(['Wysokość','Zerowa akcja','PPO best','PPO latest'],comparison,'32 sparowane scenariusze na wysokość; seed 20261024')}
<p>Łącznie: zero 4/128 (3,1%), best 34/128 (26,6%), latest 33/128 (25,8%). Różnica między checkpointami to jeden epizod. W pojedynczych nominalnych próbach wszystkie trzy warianty przechodzą cztery wysokości, w tym 2,5 m.</p>
<p>Te wyniki nie dowodzą wyższości PPO nad optymalizacją bayesowską. Pokazują wkład polityki względem jej stałej bazy PD. Docelowo interesuje nas zakres warunków, w którym PPO osiąga 70–80% poprawnych lądowań, oraz wpływ wysokości, kąta i prędkości.</p>
<h3>Nominalne lądowanie z 2,5 m</h3>
{videos(stage.get('videos',[]))}
<p class="caption">Pojedyncze próby, bez zaburzeń. Nie zastępują statystyki z tabeli. Są to wagi drugiego treningu — nie v3.</p>
<p><a href="../landing_seed606_replay_comparison_2026-10-24.html">Pełne porównanie checkpointów i dane źródłowe ↗</a></p>
<details><summary>Historyczny zapis zakończonego etapu</summary>{paragraphs([stage['summary']])}<h4>Wykonane</h4>{bullets(stage['done'])}<h4>Zmiany</h4>{bullets(stage['changed'])}<h4>Otwarte kwestie i ówczesny plan</h4>{bullets(stage['problems']+stage['next'])}</details></section>''')

    stage=stages[1]
    literature=[
        ['Mini Cheetah','Model ~1,5 m; hardware ~3,05 m nad matami','Pitch −90…+90°','Zakres v_xy: NR; reorientacja w locie'],
        ['Reactive Landing Controller','Osobne testy 0,60 i 0,80 m','Roll −40…+35°, pitch −45…+15° w teście jednej osi','Sweep |v_xy| 0–4 m/s; opisany sukces do ~3 m/s'],
        ['Guardians as You Fall','Platforma 0,4 m; inny test: najniższy element 0,1 m','Hardware pitch 45°, 75°, 105°, 135°','W dropach v=0; 1–5 m/s dotyczy uderzającej piłki'],
        ['SafeFall','Upadki w lokomocji; hardware m.in. platforma 0,30 m','Stany z trajektorii upadku; jeden zakres: NR','Perturbacje tułowia x ±2, y ±1 m/s'],
        ['Robot Crash Course','Brak jednej wysokości dropu: NR','Roll/pitch ±30°','Linear root velocity ±2 m/s; nie tylko |v_xy|'],
        ['Curriculum Learning for Landing…','NR','NR','Pełny tekst niezweryfikowany'],
    ]
    factor=data['factorization']
    explanation=''.join(f'<h4>{esc(s["title"])}</h4>{paragraphs(s["paragraphs"])}' for s in factor['sections'])
    changes=''
    for i,item in enumerate(stage['changed'],1):
        title,sep,body=item.partition(': ')
        if not sep:title,body=f'Zmiana {i}',item
        changes+=f'<h4>{esc(title.capitalize())}</h4><p>{prose(body)}</p>'
    sections.append(f'''<section id="e2" class="stage"><div class="stage-label">Etap 02 / 25.10.2026</div>
<h2>Warunki badań i rozdzielony trening v3</h2>
<p class="lead">Wysokość, orientacja i prędkość to trzy różne osie zadania. V3 rozdziela ich dobór i pomiar, zachowując tę samą politykę PPO oraz bazę PD.</p>
<p class="annotation">Ten etap opisuje przygotowaną konfigurację. Pierwszy przebieg ujawnił później błąd aktualizacji curriculum — <a href="#e3">diagnoza i poprawka są w Etapie 3</a>. Zakresy samplera i prawidłowe działanie ich adaptacji to odrębne kwestie.</p>
<nav class="section-links" aria-label="Treść etapu 2"><a href="#literatura">Warunki literatury</a><a href="#faktoryzacja">Czym jest faktoryzacja?</a><a href="#zmiany-v3">Dokładne zmiany</a><a href="#ograniczenia-v3">Ograniczenia</a></nav>
<div id="literatura"><h3>Co robiły inne prace?</h3>
{table(['Praca','Wysokość / scenariusz','Orientacja','Prędkość / interpretacja'],literature)}
<p class="caption">NR = brak zweryfikowanej wartości, nie zero. Granice jednej osi nie tworzą wspólnego obszaru gwarantowanej skuteczności. Wysokość platformy, bazy i najniższego elementu nie są zamienne.</p>
<p><a href="../article/literature-conditions.md">Pełna tabela warunków, źródła i lokalizacje w publikacjach ↗</a></p>
<details><summary>Szczegółowe omówienie i interpretacja literatury</summary>{paragraphs(stage['done'])}</details></div>
<div id="faktoryzacja" class="focus-section"><div class="stage-label">Wyjaśnienie metody</div><h3>{esc(factor['title'])}</h3><p class="lead">{esc(factor['lead'])}</p>
{table(['Grupa','Udział resetów','Przechył','Prędkość XY'],[['Nominalna','20%','Zero','Zero'],['Kątowa','20%','Losowy','Zero'],['Prędkościowa','20%','Zero','Losowa'],['Łączona','40%','Losowy','Losowa']])}
<p class="equation">Wysokość: 25% dokładnie 2,5 m + 75% losowo z [1; 2,5] m<br>Kąt: A = 1° + 2° × d<sub>A</sub><br>Prędkość: V = 0,5 + 2,5 × d<sub>V</sub> m/s na składową</p>
{explanation}</div>
<details id="zmiany-v3"><summary>Pełny opis zmian implementacyjnych v2 → v3</summary>{changes}</details>
<div id="ograniczenia-v3"><h3>Ograniczenia i dalsza ocena</h3>{paragraphs(stage['problems'])}
<details><summary>Plan operacyjny po przygotowaniu v3</summary>{paragraphs(stage['next'])}</details></div>
<p><a href="../experiments/landing_seed606_factorized_README.md">Konfiguracja i instrukcja uruchomienia v3 ↗</a></p></section>''')

    stage=stages[2]
    sections.append(f'''<section id="e3" class="stage"><div class="stage-label">Etap 03 / 26.10.2026</div>
<h2>Pierwszy log v3: diagnoza i naprawa</h2>
<p class="lead">Zadanie 9306096 losowało wysokości i grupy, ale nie zwiększało trudności kąta ani prędkości. Ten etap dokumentuje wykrycie i lokalną weryfikację poprawki. Późniejszy, pełny trening poprawionego kodu opisano w Etapie 4.</p>
{metrics(stage['metrics'])}
<h3>Co faktycznie pokazał log?</h3>
<p>Do 10:47:36 wykonano 4 063 232 przejścia i 31 699 epizodów. Proporcje grup były bliskie 20/20/20/40%. Nie było sygnałów OOM, tracebacków ani NaN/Inf. Mimo sukcesów w łatwych epizodach obie nowe EMA i oba liczniki pozostawały zerowe.</p>
{table(['Warunek ewaluacji','1 m','1,5 m','2 m','2,5 m','Średnia'],[['Nominalny','100%','100%','100%','100%','100%'],['Tylko kąty ±3°','100%','100%','100%','100%','100%'],['Tylko vx/vy ±3 m/s','12,50%','6,25%','7,81%','7,81%','8,59%'],['Kąty i prędkość','9,38%','6,25%','6,25%','6,25%','7,03%']],'Ewaluacja przy 3 145 728 przejściach; po 64 epizody na wysokość i warunek')}
<p>Na tym etapie głównym ograniczeniem jest prędkość pozioma. Wynik nie dowodzi jeszcze, że przyczyną jest brak jej pomiaru u aktora. Wysokie sukcesy treningowe dotyczą małych zakresów i nie są sprzeczne z niską skutecznością przy pełnych zaburzeniach.</p>
<h3>Przyczyna błędu</h3><p>Finalizacja lądowania w funkcji nagrody aktualizowała własny, starszy mechanizm curriculum. Omijała updater nowych czynników podłączony do inicjalizacji stanu. Następny reset pobierał nadal zerowe poziomy, więc aktywne grupy pozostawały przy ±1° i ±0,5 m/s. Wcześniejsze testy bezpośredniego updatera oraz próby PPO 16/32 przejść nie obejmowały całej ścieżki zakończonego epizodu.</p>
<h3>Co naprawiono i sprawdzono?</h3><p>Wynik z finalizacji przekazywany jest do odpowiednich updaterów. Testy obejmują reset, fizykę, reward i zakończenie lądowania; sprawdzają izolację grup oraz wzrost trudności po serii sukcesów. Po 60 rzeczywistych epizodach kątowych licznik kąta wynosi 60 i jego trudność rośnie, podczas gdy licznik prędkości pozostaje zerowy.</p>
<p>Przeszło 77 unikalnych testów i próba PPO 512 przejść obejmująca całe epizody. Naprawiono także wcześniejsze wliczanie nominalnego replay v2 do starej EMA. Historyczne pomiary checkpointów v2 nadal są pomiarami tych wag; korekty wymaga opis ich curriculum.</p>
<h3>Co dalej z zadaniem 9306096?</h3><p>Rekomendacja po analizie: zachować je jako diagnostyczną próbę małych, stałych zaburzeń i ponowić trening po przeniesieniu poprawionego kodu. W tej sesji nie zatrzymano zadania klastra i nie potwierdzono startu nowego przebiegu. Samo podmienienie plików nie naprawia już działających procesów.</p>
<p><a href="../article/v3-initial-log-9306096.md">Pełna diagnoza, wyniki testów i instrukcja po poprawce ↗</a></p>
<details><summary>Historia zapisu etapu</summary>{paragraphs([stage['summary']])}<h4>Wykonane</h4>{bullets(stage['done'])}<h4>Zmiany</h4>{bullets(stage['changed'])}<h4>Otwarte kwestie</h4>{bullets(stage['problems'])}<h4>Następne kroki</h4>{bullets(stage['next'])}</details></section>''')

    completed=data.get('completed_v3')
    if completed:
        final=completed['final_metrics']
        eval_rows=[]
        for condition,label in [('nominal','Nominalnie'),('angles','Tylko kąty'),('velocity','Tylko prędkość'),('perturbed','Kąty i prędkość')]:
            eval_rows.append([label]+[f'{100*final[f"eval/{condition}/policy/h_{h}/success_rate"]:.1f}%' for h in ['1','1p5','2','2p5']]+[f'{100*final[f"eval/{condition}/policy/success_rate"]:.1f}%'])
        previous=json.loads((ROOT/'docs/artifacts/seed606_replay_comparison_2026-10-24/checkpoints.json').read_text())['best']['metrics']
        compare=[]
        for tag,label in [('1','1 m'),('1p5','1,5 m'),('2','2 m'),('2p5','2,5 m')]:
            compare.append([label,f'{100*previous[f"eval/perturbed/policy/h_{tag}/success_rate"]:.1f}%',f'{100*final[f"eval/perturbed/policy/h_{tag}/success_rate"]:.1f}%'])
        sections.append(f'''<section id="e4" class="stage"><div class="stage-label">Etap 04 / 26.10.2026 / zadanie 9326190</div>
<h2>Naprawiony v3 zakończył pełny trening</h2>
<p class="lead">30 015 488 przejść i 257 537 epizodów, zakończenie o 15:48:39. Oba curricula pracują; końcowa ewaluacja obejmuje rzeczywiste końcowe wagi. Nominalne lądowania są zachowane, ale przy pełnych zaburzeniach wynik nadal wynosi około jednej trzeciej prób.</p>
{metrics(stages[3]['metrics'])}
<h3>Co pokazuje końcowa ewaluacja?</h3>
{table(['Warunek','1 m','1,5 m','2 m','2,5 m','Średnia PPO'],eval_rows,'64 epizody na wysokość i warunek; pełny cel ±3° i vx/vy ±3 m/s')}
<p>Nominalnie i przy samych małych przechyłach końcowa polityka przechodzi wszystkie próby. Sama prędkość daje 88/256 sukcesów (34,4%), a połączenie kąta i prędkości 85/256 (33,2%). Przy 2,5 m połączone zaburzenia dają 18/64, czyli 28,1%. Najczęstszą przyczyną porażki jest niedozwolony kontakt; przy 2,5 m pojawiają się też pojedyncze base crash. Z logu nie można ustalić dokładnego collidera — do tego potrzebne jest odtworzenie checkpointu.</p>
<h3>Wzrost trudności działa, ale prędkość zatrzymała postęp</h3>
<figure class="chart"><img src="../artifacts/v3_completed_9326190_2026-10-26/learning_curves.svg" alt="Krzywe ewaluacji: nominalnie około 100%, prędkość i połączone zaburzenia rosną do około 33%; trudność kąta osiąga 1, prędkości około 0,44."><figcaption>Góra: stałe pełne zakresy ewaluacji. Dół: średnie poziomy curriculum z rolloutów. Zacienienie 70–80% oznacza cel, nie osiągnięty wynik.</figcaption></figure>
<p>Trudność kąta osiągnęła 1, czyli pełne ±3°. Średnia trudność prędkości to około 0,441; we wzorze V=0,5+2,5d odpowiada to zakresowi około ±1,60 m/s na składową, podczas gdy ewaluacja pozostaje przy ±3 m/s. Nie oznacza to, że każdy worker miał identyczny poziom — jest to średnia. Około 10 mln przejść kąt był już blisko pełnego zakresu, a prędkość dopiero przy poziomie 0,27.</p>
<p>Końcowe sukcesy rolloutów: nominal 99,6%, angles 99,2%, velocity 74,4%, combined 72,3%. To wyniki przy aktualnym rozkładzie treningowym, a nie przy pełnym zakresie testowym. Nie można uznać ich za realizację celu 70–80% w domenie docelowej. W logu nie ma OOM, tracebacków ani NaN; KL około 0,078 i clip fraction około 0,50 sugerują jednak większe zmiany polityki niż w poprzednim przebiegu. To przesłanka do kontroli aktualizacji, nie dowód rozbiegania treningu.</p>
<h3>Jak wypada to względem v2?</h3>
{table(['Wysokość','V2 best · 23,07 mln','V3 końcowy · 30,02 mln'],compare,'Ta sama walidacyjna rodzina scenariuszy z łączonymi zaburzeniami; nie nowy zamknięty test')}
<p>Best v2 miał średnio 35,9% i najgorszą wysokość 23,4%; końcowy v3 ma średnio 33,2% i najgorszą wysokość 28,1%. V3 jest więc bardziej wyrównany po wysokościach, ale nie ma wyższej średniej. To odpowiada nowemu kryterium wyboru, które preferuje najgorszą wysokość. Jeden seed i zestaw używany do selekcji nie dowodzą przewagi metody. Nie porównujemy tych liczb bezpośrednio z wcześniejszym testem 32 nowych scenariuszy na wysokość — to inny zestaw.</p>
<h3>Best, latest i dalsza decyzja</h3>
<p>Kryterium selekcji w logu po raz pierwszy osiąga najwyższy wynik przy 27 262 976 przejściach. Końcowa ewaluacja ma ten sam wynik najgorszej wysokości i tę samą średnią; przy zapisie tylko ścisłej poprawy best powinien pozostać wcześniejszy. To wniosek z metryk i kodu, wymagający potwierdzenia metadanymi pobranych plików. Najpierw porównamy best i latest na nowych stanach oraz odtworzymy porażki.</p>
<p>Nie uruchamiałbym od razu kolejnego identycznego długiego treningu. Następne badanie to mapa modułu i kierunku prędkości przy kontrolowanej wysokości oraz osobna ablacja informacji o v_xy. Obecny aktor nadal nie ma bezpośredniej prędkości liniowej. Czy to główna przyczyna ograniczenia, musi rozstrzygnąć eksperyment, a nie samo niskie success rate.</p>
<p class="caption">Pełna mapa czynników, kolejne seedy i zamknięty test nadal są otwarte; postęp checklisty pozostaje 4/8. Zakończenie jednego przebiegu nie zastępuje tych zadań.</p>
<p><a href="../artifacts/v3_completed_9326190_2026-10-26/parsed.json">Wszystkie sparsowane rollouty, ewaluacje i hash logu ↗</a></p></section>''')

    checkpoint_experiments=data.get('checkpoint_experiments')
    if checkpoint_experiments:
        from experiments.reporting.progress_v3_experiment_section import build
        sections.append(build(table,videos))

    if data.get('tilt_diagnostic'):
        from experiments.reporting.progress_tilt_section import build as build_tilt
        sections.append(build_tilt(table))

    if data.get('pd_stability_v4'):
        from experiments.reporting.progress_pd_stability_section import build as build_pd_stability
        sections.append(build_pd_stability(table))

    if data.get('v4_completed'):
        from experiments.reporting.progress_v4_completed_section import build as build_v4_completed
        sections.append(build_v4_completed(table,videos))

    if data.get('velocity_research_plan'):
        from experiments.reporting.progress_velocity_plan_section import build as build_velocity_plan
        sections.append(build_velocity_plan(table,videos))

    if data.get('velocity_training_v5'):
        from experiments.reporting.progress_velocity_v5_section import build as build_velocity_v5
        sections.append(build_velocity_v5(table))

    if data.get('v5_completed'):
        from experiments.reporting.progress_v5_completed_section import build as build_v5_completed
        sections.append(build_v5_completed(table,videos))

    if data.get('v6_prepared'):
        from experiments.reporting.progress_v6_preparation_section import build as build_v6_preparation
        sections.append(build_v6_preparation(table))

    if data.get('v6_completed'):
        from experiments.reporting.progress_v6_completed_section import build as build_v6_completed
        sections.append(build_v6_completed(table,videos))
    if data.get('v7_prepared'):
        from experiments.reporting.progress_v7_preparation_section import build as build_v7_preparation
        sections.append(build_v7_preparation(table))

    if data.get('v7_completed'):
        from experiments.reporting.progress_v7_completed_section import build as build_v7_completed
        sections.append(build_v7_completed(table,videos))

    if data.get('v8_prepared'):
        from experiments.reporting.progress_v8_preparation_section import build as build_v8
        sections.append(build_v8(table))

    css='''
:root{color-scheme:dark;--bg:#11171e;--fg:#dce4ec;--muted:#a3b1bf;--line:#34404b;--accent:#84d4c0;--warm:#edbe85;--surface:#18212a}
*{box-sizing:border-box}html{scroll-padding-top:88px}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.8 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:4px}a:hover{color:#b4f1df}a:focus-visible,summary:focus-visible,video:focus-visible{outline:2px solid var(--accent);outline-offset:5px}p,li{max-width:82ch;overflow-wrap:anywhere}p{margin:0 0 22px}strong{color:#f3f6fa}h1,h2,h3,h4{color:#f1f5fa;line-height:1.25;font-weight:620;letter-spacing:-.025em}h1{font-size:clamp(38px,5vw,66px);max-width:18ch;margin:22px 0 28px}h2{font-size:clamp(30px,3vw,42px);margin:14px 0 26px;max-width:25ch}h3{font-size:27px;margin:44px 0 20px}h4{font-size:21px;margin:32px 0 15px}ul{padding-left:24px}li{padding-left:4px;margin:10px 0}.container{width:min(1120px,calc(100% - 88px));margin:auto}.masthead{padding:64px 0 40px}.kicker,.stage-label{font-size:12px;font-weight:650;text-transform:uppercase;letter-spacing:.12em;color:var(--accent)}.intro{font-size:21px;color:var(--muted);max-width:65ch}.project-progress{display:flex;flex-wrap:wrap;gap:10px 32px;color:var(--muted);font-size:14px;margin:30px 0 0;padding-top:20px;border-top:1px solid var(--line)}.project-progress b{color:var(--fg)}.navigation{position:sticky;top:0;z-index:5;background:rgba(17,23,30,.97);border-block:1px solid var(--line)}.navigation .container{display:flex;align-items:center;gap:28px;overflow:auto;white-space:nowrap}.navigation a{padding:16px 0;color:var(--muted);font-size:14px;text-decoration:none;border-bottom:2px solid transparent}.navigation a:hover,.navigation a[aria-current=true]{color:var(--accent);border-bottom-color:var(--accent)}.stage{padding:76px 0;border-bottom:1px solid var(--line)}.lead{font-size:20px;color:#c8d4df;max-width:73ch}.annotation{padding:4px 0 4px 22px;border-left:2px solid var(--warm);color:#cfbea7}.caption{font-size:14px;color:var(--muted);max-width:90ch}.section-links{display:flex;gap:12px 28px;flex-wrap:wrap;font-size:14px;margin:30px 0}.section-links a{text-decoration:none}.metrics{display:flex;gap:24px 56px;flex-wrap:wrap;margin:30px 0 40px;padding:20px 0;border-block:1px solid var(--line)}.metrics div{display:flex;flex-direction:column-reverse}.metrics dt{font-size:13px;color:var(--muted)}.metrics dd{margin:0;font-size:30px;font-variant-numeric:tabular-nums;color:#f3f6fa}.table-scroll{overflow:auto;margin:25px 0;max-width:100%}table{width:100%;border-collapse:collapse;font-size:15px;line-height:1.6}th,td{text-align:left;vertical-align:top;padding:15px 15px 15px 0;border-bottom:1px solid var(--line);min-width:82px}th{color:var(--muted);font-weight:550;font-size:13px}td:first-child{color:#f1f5fa;font-weight:550}caption{text-align:left;color:var(--muted);font-size:13px;padding-bottom:12px}.films{display:grid;grid-template-columns:1fr 1fr;gap:32px;margin:24px 0}figure{margin:0;min-width:0}video{width:100%;height:auto;aspect-ratio:16/10;background:#080c10;display:block}figcaption{font-size:14px;line-height:1.6;margin-top:15px}figcaption strong,figcaption span{display:block}figcaption span{color:var(--muted);margin:7px 0}.download{font-size:13px}details{margin:30px 0;padding:18px 0;border-block:1px solid var(--line)}summary{cursor:pointer;color:var(--accent);font-size:16px}details[open]>summary{margin-bottom:24px}.focus-section{margin:64px 0;padding-top:36px;border-top:1px solid var(--accent)}.focus-section>h3{font-size:32px;margin-top:14px}.equation{font:18px/2.1 ui-monospace,SFMono-Regular,Consolas,monospace;color:#b3e6d9;border-left:2px solid var(--accent);padding:8px 0 8px 24px;margin:32px 0;max-width:100%;overflow-wrap:anywhere}.status-line{margin-top:35px}.status-line strong{color:var(--warm)}.footer{padding:40px 0 64px;color:var(--muted);font-size:14px}.footer a{margin-right:20px}.skip{position:absolute;top:-100px}.skip:focus{top:10px;left:10px;z-index:10;background:var(--bg);padding:10px}
@media(max-width:720px){body{font-size:16px}.container{width:calc(100% - 36px)}.masthead{padding:34px 0 26px}.navigation .container{gap:22px}.navigation a{font-size:13px;padding:13px 0}.stage{padding:46px 0}.films{grid-template-columns:1fr;gap:28px}h3{font-size:24px}.lead{font-size:18px}.intro{font-size:18px}.metrics{gap:20px 30px}.metrics dd{font-size:26px}.focus-section>h3{font-size:28px}th,td{min-width:106px}.equation{font-size:15px;padding-left:14px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{:root{--bg:white;--fg:#152030;--muted:#465568;--line:#ccd2d8;--accent:#126459;--warm:#784c22}body{font-size:11pt}h1,h2,h3,h4,strong,td:first-child{color:#152030}.container{width:100%}.navigation{display:none}.stage{padding:25px 0}video{display:none}.films{display:block}figure{margin-bottom:20px}.annotation{color:#574737}.stage-label,.kicker{color:#126459}}
'''
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    result=f'''<!doctype html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><title>Emergency landing z PPO — raport badań</title><style>{css}</style></head><body>
<a class="skip" href="#aktualnie">Przejdź do treści</a><header class="masthead container"><div class="kicker">RL-X / Raport badań / 27.10.2026</div><h1>Emergency landing z polityką PPO</h1><p class="intro">Od przygotowania regulatora do wyznaczenia zakresu skutecznego lądowania. Wysokość, orientacja i prędkość początkowa jako osobne osie eksperymentu.</p><div class="project-progress"><span><b>50%</b> · 4 z 8 kamieni milowych checklisty</span><span>Cel: <b>70–80% skuteczności</b> w określonej domenie</span><span>Aktualizacja: 27 października 2026</span></div></header>
<nav class="navigation" aria-label="Nawigacja raportu"><div class="container"><a href="#aktualnie">Aktualny stan</a><a href="#e1">01 · PD i PPO</a><a href="#e2">02 · Warunki i v3</a><a href="#faktoryzacja">Faktoryzacja</a><a href="#e3">03 · Diagnoza błędu</a><a href="#dokumenty">Dokumenty</a></div></nav>
<main class="container"><section id="aktualnie" class="stage"><div class="stage-label">Stan na podstawie ostatniego przeanalizowanego logu</div><h2>Najpierw poprawne curriculum, potem ocena v3</h2><p class="lead">Pierwszy przebieg v3 pozostał przy małych, stałych zaburzeniach z powodu błędu aktualizacji. Przyczyna jest ustalona, poprawka przeszła testy pełnych epizodów. Nie mamy jeszcze wyników pełnego poprawionego treningu.</p><p class="status-line"><strong>Wniosek operacyjny:</strong> zachować zadanie 9306096 jako przebieg diagnostyczny, a właściwy v3 uruchomić po przeniesieniu poprawionego kodu. Raport nie potwierdza zatrzymania starego ani startu nowego zadania.</p><p><a href="#e3">Przeczytaj diagnozę</a> · <a href="#faktoryzacja">Zrozum rozdzielenie warunków treningu</a></p></section>
{''.join(sections)}
<section id="dokumenty" class="stage"><div class="stage-label">Materiały i odtwarzalność</div><h2>Dane, konfiguracje i plan artykułu</h2><ul><li><a href="../article/literature-conditions.md">Warunki eksperymentów w literaturze i źródła pierwotne</a></li><li><a href="../article/implementation-agenda.md">Agenda eksperymentów do artykułu</a></li><li><a href="../experiments/landing_seed606_factorized_README.md">Instrukcja i parametry treningu v3</a></li><li><a href="../article/v3-initial-log-9306096.md">Pełna diagnoza pierwszego v3</a></li><li><a href="../artifacts/v3_initial_9306096_2026-10-26/parsed.json">Sparsowany log v3 i hash źródła</a></li><li><a href="../artifacts/seed606_replay_comparison_2026-10-24/heldout_results.json">Wyniki nowych scenariuszy best/latest/zero</a></li><li><a href="report-data.json">Zachowane dane i historia raportu</a></li></ul><p class="caption">Daty i wyniki odnoszą się do wskazanych eksperymentów. Zmiana układu raportu nie jest nowym treningiem ani wzrostem postępu projektu. Szczegółowe dane wszystkich trzech etapów zachowano.</p></section></main>
<footer class="footer container">Silver Badger · Emergency landing · PPO z residualnymi celami pozycji stawów</footer>
<script type="application/json" id="report-data">{payload}</script>
<script>const links=[...document.querySelectorAll('.navigation a')]; const observer=new IntersectionObserver(entries=>{{for(const e of entries){{if(e.isIntersecting){{for(const a of links){{if(a.hash==='#'+e.target.id)a.setAttribute('aria-current','true');else a.removeAttribute('aria-current');}}}}}}}},{{rootMargin:'-12% 0px -65% 0px'}}); for(const a of links){{const node=document.querySelector(a.hash);if(node)observer.observe(node);}} function revealHash(){{if(!location.hash)return;const node=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(!node)return;const detail=node.closest('details');if(detail)detail.open=true;node.scrollIntoView({{behavior:'instant',block:'start'}});}} window.addEventListener('load',revealHash);window.addEventListener('hashchange',revealHash);</script></body></html>'''
    if completed:
        overview='''<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan / pełny log 9326190</div><h2>V3 działa poprawnie. Prędkość nadal ogranicza lądowanie.</h2><p class="lead">Naprawiony trening zakończył 30,02 mln przejść. Końcowa polityka osiąga 100% nominalnie i przy samych kątach, 34,4% przy samej prędkości oraz 33,2% przy obu zaburzeniach. Najgorsza wysokość przy łączonych zaburzeniach: 28,1% z 2,5 m.</p><p class="status-line"><strong>Co to oznacza:</strong> zachowaliśmy nominalne lądowanie i uzyskaliśmy bardziej wyrównane wyniki po wysokościach. Cel 70–80% w pełnej domenie nie został osiągnięty. Kolejny krok to odtworzenie checkpointów i diagnoza prędkości, nie powtórzenie tego samego treningu w ciemno.</p><p><a href="#e4">Przejdź do analizy zakończonego v3</a> · <a href="#faktoryzacja">Czym jest faktoryzacja?</a></p></section>'''
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:overview,result,count=1,flags=re.S)
        result=result.replace('<a href="#dokumenty">Dokumenty</a>','<a href="#e4">04 · Wynik v3</a><a href="#dokumenty">Dokumenty</a>')
        result=result.replace('dane wszystkich trzech etapów','dane wszystkich czterech etapów')
    if checkpoint_experiments:
        overview='''<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan / eksperymenty na checkpointach v3</div><h2>Wysoka skuteczność przy małych prędkościach. Pełny zakres pozostaje trudny.</h2><p class="lead">Po 1968 ocenianych epizodach oba checkpointy zachowują nominalne lądowania. W nowych próbach do 1 m/s z przechyłem ±3° best osiąga 97,7%, latest 99,2%. Przy pełnych vx/vy ±3 m/s wyniki spadają do 24,6% i 21,5%.</p><p class="status-line"><strong>Główny wynik:</strong> wyznaczyliśmy użyteczny ograniczony obszar działania, ale nie osiągnęliśmy celu 70–80% w pierwotnej pełnej domenie. Sam dobór checkpointu nie rozwiązuje ograniczenia prędkości. Mały obszar był badany eksploracyjnie po screeningu.</p><p><a href="#e5">Eksperymenty best/latest i filmy</a> · <a href="#faktoryzacja">Wyjaśnienie faktoryzacji</a></p></section>'''
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:overview,result,count=1,flags=re.S)
        result=result.replace('<a href="#dokumenty">Dokumenty</a>','<a href="#e5">05 · Eksperymenty wag</a><a href="#dokumenty">Dokumenty</a>')
        result=result.replace('dane wszystkich czterech etapów','dane wszystkich pięciu etapów')
    if data.get('tilt_diagnostic'):
        overview='<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan / test przechyłów i dynamiki</div><h2>Większe przechyły ujawniły oscylacje PD zależne od kroku fizyki.</h2><p class="lead">W 204 nowych próbach oba modele przechodzą 14/16 lądowań przy 10°, lecz tylko 1/16 i 3/16 przy 30°. Reagują przed kontaktem. Rejestracja 5 ms wykryła naprzemienne nasycenie napędów łydki; w krótkim replayu tych samych komend przy 2,5 i 1,25 ms nasycenie znika.</p><p>Przed kolejnym treningiem potrzebna jest kontrola zbieżności dynamiki PD i kontaktu. Replay nie dowodzi jeszcze poprawy skuteczności lądowania.</p><p><a href="#e6">Test przechyłów i wykresy</a> · <a href="#e5">Wcześniejsze porównanie best/latest/PD</a> · <a href="#zakres-stawow">Zakres ruchu stawów</a></p></section>'
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:overview,result,count=1,flags=re.S)
        result=result.replace('<a href="#dokumenty">Dokumenty</a>','<a href="#e6">06 · Przechyły i PD</a><a href="#dokumenty">Dokumenty</a>')
        result=result.replace('dane wszystkich pięciu etapów','dane wszystkich sześciu etapów')
    if data.get('pd_stability_v4'):
        overview='<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan / przygotowany trening v4</div><h2>Oscylacje usunięte w badanych przebiegach. Nowy trening użyje stabilniejszej fizyki.</h2><p class="lead">Mniejsze Kd usuwa oscylacje, ale nie daje wyraźnej poprawy lądowań istniejących wag. V4 zachowuje oryginalny PD i używa implicitfast z krokiem 2,5 ms. Sprawdzono również 1,25 ms oraz zgodność CPU search–PPO przez kontakt.</p><p>test2.sh jest przygotowany do świeżego treningu v4. Smoke 512 przejść potwierdził uruchomienie fizyki v4; nowe zakresy przeszły test resetów i konfiguracji. Pełny trening na klastrze pozostaje do uruchomienia. Rewizja 27.10: curriculum i ewaluacja do 2,2 m, roll/pitch docelowo ±10° na oś. std pozostaje 0,15. Zmienia się też domena zadania względem v3.</p><p><a href="#e7">Eksperymenty z gainami i konfiguracja v4</a> · <a href="#e6">Diagnoza przechyłów</a> · <a href="#e5">Wcześniejsze wyniki polityk</a></p></section>'
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:overview,result,count=1,flags=re.S)
        result=result.replace('<a href="#dokumenty">Dokumenty</a>','<a href="#e7">07 · Stabilność i v4</a><a href="#dokumenty">Dokumenty</a>')
        result=result.replace('dane wszystkich sześciu etapów','dane wszystkich siedmiu etapów')
    if data.get('v4_completed'):
        overview='<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan / zakończony trening v4 · 9376691</div><h2>Przechyły ±10° opanowane. Pełna domena nadal poniżej celu.</h2><p class="lead">V4 zakończył 30 mln przejść. Nowy niezależny test: 128/128 sukcesów przy samych kątach, 107/256 (41,8%) przy kątach i pełnych vx/vy ±3 m/s. Przy ograniczeniu modułu prędkości do 1 m/s i kątach ±10°: 128/128. Best i latest mają identyczne wagi polityki.</p><p>Wcześniejsze naprzemienne nasycenie łydek nie powróciło w zapisanej diagnostyce. Głównym ograniczeniem pozostaje prędkość: siatka bez przechyłu daje 28/32 sukcesy przy 2 m/s, ale tylko 6/32 przy 2,5 m/s. Wyników v3 i v4 nie traktujemy jako izolowanej ablacji integratora, bo zmieniła się też domena treningu.</p><p><a href="#e8">Etap 08: pełne wyniki, mapy i dwa filmy porównawcze →</a></p><p class="caption">Cel 70–80% w pełnej domenie nie jest jeszcze osiągnięty. Wybierz etap w bocznym pasku; każda zakładka zachowuje historyczne wyniki i dostępne nagrania.</p></section>'
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:overview,result,count=1,flags=re.S)
        result=result.replace('dane wszystkich siedmiu etapów','dane wszystkich ośmiu etapów')
    if data.get('velocity_research_plan'):
        result=result.replace('<p><a href="#e8">Etap 08: pełne wyniki, mapy i dwa filmy porównawcze →</a></p>','<p><a href="#e8">Etap 08: wyniki i filmy</a> · <a href="#e9">Etap 09: plan po konsultacji z promotorem — prędkość w obserwacji →</a></p>')
        result=result.replace('dane wszystkich ośmiu etapów','dane wszystkich dziewięciu etapów')
    if data.get('velocity_training_v5'):
        overview='<section id="aktualnie" class="stage"><div class="stage-label">Aktualny stan / przygotowany trening v5 z prędkością</div><h2>Polityka otrzymuje prędkość. V5 jest gotowy do uruchomienia.</h2><p class="lead">Docelowe źródło prędkości na robocie to OptiTrack. Aktor v5 ma dwa dodatkowe kanały: przód i bok, w poziomej ramie kierunku bazy. test2.sh uruchamia świeże wagi na tej samej fizyce i zadaniu co v4.</p><p>Przeszło 41 testów oraz krótki trening 1024 przejść z końcową sparowaną ewaluacją 64 epizodów. Pełny v5 na klastrze pozostaje do uruchomienia. Dostępna jest kontrola masked z identyczną architekturą. Obecne pełne wyniki 41,8% przy połączonych zaburzeniach dotyczą jeszcze v4.</p><p><a href="#e10">Etap 10: gotowy trening v5 i kontrakt OptiTrack →</a> · <a href="#e9">Plan eksperymentów</a> · <a href="#e8">Ostatni zakończony trening i filmy</a></p></section>'
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:overview,result,count=1,flags=re.S)
        result=result.replace('dane wszystkich dziewięciu etapów','dane wszystkich dziesięciu etapów')
    if data.get('v5_completed'):
        from experiments.reporting.progress_v5_completed_section import overview as v5_overview
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:v5_overview(),result,count=1,flags=re.S)
        result=result.replace('dane wszystkich dziesięciu etapów','dane wszystkich jedenastu etapów')
        result=result.replace('27.10.2026</div><h1>','28.10.2026</div><h1>').replace('Aktualizacja: 27 października 2026','Aktualizacja: 28 października 2026')
    if data.get('v6_prepared'):
        from experiments.reporting.progress_v6_preparation_section import overview as v6_overview
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:v6_overview(),result,count=1,flags=re.S)
        result=result.replace('dane wszystkich jedenastu etapów','dane wszystkich dwunastu etapów')
    if data.get('v7_prepared'):
        from experiments.reporting.progress_v7_preparation_section import overview as v7_overview
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:v7_overview(),result,count=1,flags=re.S)
        result=result.replace('dane wszystkich dwunastu etapów','dane wszystkich czternastu etapów')
        result=result.replace('28.10.2026</div><h1>','29.10.2026</div><h1>').replace('Aktualizacja: 28 października 2026','Aktualizacja: 29 października 2026')
    if data.get('v7_completed'):
        from experiments.reporting.progress_v7_completed_section import overview as v7_completed_overview
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:v7_completed_overview(),result,count=1,flags=re.S)
        result=result.replace('dane wszystkich czternastu etapów','dane wszystkich piętnastu etapów')
    if data.get('v8_prepared'):
        from experiments.reporting.progress_v8_preparation_section import overview as v8_overview
        result=re.sub(r'<section id="aktualnie".*?</section>',lambda _:v8_overview(),result,count=1,flags=re.S)
        result=result.replace('dane wszystkich piętnastu etapów','dane wszystkich szesnastu etapów')
    result=result.replace('</style>','figure.chart{margin:28px 0}figure.chart img{display:block;width:100%;height:auto}figure.chart figcaption{color:var(--muted)}</style>',1)
    from experiments.reporting.report_side_tabs import apply_side_tabs
    result=apply_side_tabs(result,data)
    (REPORT/'index.html').write_text(result)
    print(REPORT/'index.html')


if __name__=='__main__':main()
