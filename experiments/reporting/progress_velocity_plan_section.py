"""Research plan following the advisor discussion; no v5 implementation claim."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def build(table,videos):
    audit=json.loads((ROOT/'docs/artifacts/velocity_research_plan_2026-10-27/observation_audit.json').read_text())
    example=audit['examples'][1]['local_velocimeter'][0]
    return f'''<section id="e9" class="stage"><div class="stage-label">Etap 09 / 27.10.2026 / plan po konsultacji z promotorem</div>
<h2>Prędkość jako główna oś artykułu: mierzymy wpływ informacji dostępnej polityce.</h2>
<p class="lead">Plan zakłada dodanie aktorowi prędkości wzdłużnej i bocznej oraz kontrolowane porównanie braku informacji, prędkości rzeczywistej i obserwacji obarczonej błędami. W tej części zweryfikowano literaturę i kod oraz przygotowano protokół. Nowe wejścia i trening v5 nie zostały jeszcze wdrożone ani uruchomione.</p>
<h3>Co potwierdził przegląd?</h3>
<p>Skupienie na prędkości jest uzasadnione wynikami v4, ale nie ma podstaw do twierdzenia, że wysokich prędkości wcześniej nie badano. Nie wykonano systematycznej bibliometrii porównującej liczby publikacji o kątach i prędkościach. W najbliższych pracach występują bezpośrednie kontrprzykłady.</p>
{table(['Źródło pierwotne','Znaczenie dla planu'],[
['Roscia i in., Reactive Landing Controller (2023)','Modelowy landing: estymata prędkości, mapa kierunków, do 3 m/s w symulacji.'],
['Li i in., Dynamic Fall Recovery Control (2024)','RL recovery: estymator prędkości i student otrzymujący estymaty; opisano do 3 m/s.'],
['Robot Crash Course (2025)','Root linear velocity jest wejściem polityki; zadanie miękkiego upadku i docelowej pozy.'],
['Mini Cheetah / Guardians as You Fall','Silny nacisk na orientację i bezpieczny upadek; prędkość piłki w GYF nie jest prędkością robota.'],
['PPO lunar landing (2026)','Rozważa prędkości, ale przy innych skalach i grawitacji.'],
['Curriculum Learning for Landing of Quadruped Robots','Pełny tekst nadal do zdobycia; brak dostępu nie dowodzi braku badań prędkości.']])}
<p>Źródła: <a href="https://iit-dlslab.github.io/papers/roscia23ral.pdf">Reactive Landing Controller</a>, <a href="https://www.mdpi.com/2313-7673/9/4/193">Dynamic Fall Recovery Control</a>, <a href="https://arxiv.org/html/2511.10635v1">Robot Crash Course</a>, <a href="https://www.mdpi.com/2075-1702/14/4/417">PPO lunar landing</a>. Pełna tabela źródeł, miejsc w tekście i ograniczeń jest w <a href="../article/velocity-focused-plan.md">planie artykułu</a>.</p>
<p><strong>Proponowany wkład:</strong> określenie, jak dostępność i jakość obserwacji prędkości zmieniają zakres skutecznego emergency landingu PPO w funkcji wysokości, modułu i kierunku ruchu. Samo dodanie vx/vy ani pojedynczy udany film przy 3 m/s nie stanowią deklarowanej nowości.</p>
<h3>Audyt kodu: dwa kanały wymagają precyzyjnej definicji</h3>
<p>Aktor v4 ma {audit['actor_dimension']} wejść i nie widzi prędkości liniowej; critic ma {audit['critic_dimension']} wejścia i widzi trzy kanały velocimetru. goal_velocities jest komendą, a nie pomiarem rzeczywistej prędkości.</p>
<p>Obecny velocimeter mierzy prędkość punktu IMU w jego lokalnej ramie. Przy czysto pionowej prędkości świata −5 m/s i pitch=10° lokalny kanał x wynosił w audycie +{example:.3f} m/s, mimo zerowej prędkości poziomej. Bez określenia układu nie można nazwać pierwszych dwóch kanałów „poziomymi”. Site IMU jest przesunięty o 4,9 cm względem początku bazy; przy obrocie dochodzi różnica omega×r. <a href="https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-velocimeter">Semantyka velocimetru w dokumentacji MuJoCo</a>.</p>
<p>Planowane wejścia to v_forward i v_lateral początku bazy, w układzie związanym z kierunkiem robota, ale wypoziomowanym względem grawitacji. Pierwszy wariant jawnie używa prawdziwej prędkości z symulatora (oracle). Nie przedstawiamy go jako pomiaru zwykłej IMU. Aktor nie otrzymuje przy okazji wysokości ani prędkości pionowej.</p>
<h3>Kontrolowana macierz eksperymentów</h3>
{table(['Wariant','Dwa wejścia prędkości','Cel'],[
['A — masked','Stale zero; aktor ma 80 wejść','Kontrola informacyjna przy tej samej liczbie parametrów'],
['B — oracle XY','Rzeczywiste dwie prędkości; 80 wejść','Czy informacja pomaga przy wspólnym zadaniu?'],
['C — emulated estimate','Kontrolowany szum, bias, opóźnienie i sample-and-hold','Czy efekt przeżywa błędy obserwacji?'],
['D — estymator na robocie','Warunkowo: dostępne, scharakteryzowane źródło prędkości','Dopiero to może wspierać deklarację wdrożenia'],
['PD / zero akcji','Stałe home i te same gainy','Ablacja residualu i kontekst, nie główny cel artykułu']])}
<p>Critic pozostaje przy swoich dotychczasowych 94 wejściach. A i B startują od świeżych wag; v4 z 78 wejściami pozostaje referencją historyczną. W A/B zamrażamy fizykę, PD, home, reward, kryteria sukcesu, domain/curriculum, std, budżet i selekcję checkpointu. Zmiany samplera dużych prędkości albo nagrody wymagają późniejszej, osobnej ablacji.</p>
<h3>Kolejność i budżet</h3>
{table(['Etap','Praca','Warunek przejścia'],[
['P0 — implementacja','Ramy i punkt prędkości; tryby A/B; brak przecieku; metadata; testy i smoke','Dwa poprawne kanały, identyczna fizyka, brak wznowienia niezgodnych wag'],
['P1 — pilot','Po 4 194 304 przejścia A/B, seed 1','Sprawdzenie pipeline’u i trendu na walidacji, nie końcowy wynik'],
['P2 — ablacja główna','A/B × co najmniej 3 seedy × 30 015 488 przejść','6 przebiegów; wyniki na zamrożonym teście i zmienność po seedach'],
['P3 — jakość prędkości','Osobno szum 0–0,5 m/s, bias, lag 0/20/40/80 ms i odświeżanie 50/25/10 Hz','Najpierw wrażliwość B, później trening C z takimi błędami'],
['P4 — wyższe prędkości','Osobna zmiana pokrycia curriculum przy ustalonej obserwacji','Wspólna domena testowa; jawne oznaczenie zmiany wsparcia rozkładu']])}
<p>Nie zakładamy, że historia IMU po resecie z nieznanym v0 wystarczy do odzyskania stałej prędkości początkowej. Estymator wymaga inicjalizacji przed lotem lub dodatkowych informacji. Zakresy szumu są propozycją testu, nie kalibracją rzeczywistego sensora; fizyka z randomizacją wyłączoną nie może przypadkowo zerować szumu obserwacji.</p>
<h3>Co ma pokazać artykuł?</h3>
<p>Wynik główny: wspólna domena v4, wysokości 1/1,5/2/2,2 m, roll/pitch ±10°, vx/vy ±3 m/s. Proponowane 200 nowych scenariuszy na wysokość i checkpoint, z tymi samymi resetami dla metod. Dotychczasowy test jest już rozwojowy; końcowy zestaw należy zamrozić przed użyciem. Praktyczny cel: ≥80% średnio i ≥70% dla każdej wysokości, z osobno raportowaną niepewnością i zmiennością po seedach.</p>
<p>Drugim wynikiem jest mapa h × moduł × kierunek: screening do 4 m/s co 0,5 m/s, kierunki co 30°, później dodatkowe losowania przy granicy. Rozróżniamy przód/tył, bok oraz przekątne. Przy boxie vx/vy ±3 prędkość 4 m/s wzdłuż osi jest poza treningiem, ale po przekątnej może być wewnątrz — OOD określamy składowymi. Nie zastępujemy mapy pojedynczym największym udanym lądowaniem.</p>
<p>Analiza mechanizmu obejmie przygotowanie stóp przed kontaktem, czas reakcji, prędkość resztkową i drogę hamowania, śledzenie celów, nasycenia oraz przyczyny porażek. W locie nogi przygotowują kontakt; nie przypisujemy polityce aktywnego hamowania poziomego pędu CoM bez zewnętrznego kontaktu. Ucięte przebiegi porażek wymagają osobnego protokołu do oceny pełnych obciążeń.</p>
<h3>Przykład motywacyjny — nie estymata skuteczności</h3>
{videos([dict(title='2 m / moduł 3 m/s / roll=+10°, pitch=+10°',src='media/etap8-3ms-2m-10deg.mp4',note='Wybrana udana próba v4, kierunek 190°. Film potwierdza możliwość tego pojedynczego lądowania; nie dowodzi wysokiej skuteczności we wszystkich kierunkach.')])}
<p><a href="../article/velocity-focused-plan.md">Pełny plan, literatura i protokół eksperymentów →</a> · <a href="../artifacts/velocity_research_plan_2026-10-27/observation_audit.json">Audyt maski obserwacji i ramy prędkości</a></p></section>'''
