Optymalizacja korzysta z nominalnej fizyki treningu Silver Badger.

Aktualizacja 2026-10-22: **stopy i dolna 1/3 łydki mogą dotykać podłoża**.
Górne 2/3 łydki nadal ustawiają `disallowed_contact=True`, `Safe=False`
i porażkę PPO niezależnie od siły. Podział jest przy |y|=0,1077073809 m
w układzie łydki: dokładne 2/3 długości collidera, wzdłuż osi y. Górna
część ma 14,46 cm, dolna 7,23 cm. Obudowa kolana
należy do części górnej. Nazwa `RL/RR/FR/FL_calf` oznacza część górną,
a `_calf_lower` dolną. Obie części zachowują kolizję fizyczną; dolna nie
wchodzi do kar, czasu kontaktu ani limitu siły łydki i nie liczy się jako
podparcie stopą. Metryki `calf_*` i `shin_*` opisują część górną (oraz
dotychczasowe pomocnicze cylindry kolana dla `shin_*`). Reguła pochodzi
ze wspólnej mapy CPU/PPO i jest używana również w kodzie MJX.

Meshe są podziałem wcześniejszej wypukłej otoczki STL. Łydka kończy się
przy |y|=0,180 m, przed kulą stopy. Na klaster trzeba przenieść XML i
cztery nowe pliki `assets/l2{l,r}_calf_{upper,lower}_collision.obj`.
Odtwarzanie geometrii: `rlx/bin/python search/build_calf_collision_meshes.py`
(wymaga dwóch źródłowych `l2{l,r}_calf_collision.obj`). Masy, PD, home,
krok i parametry kontaktu nie zostały zmienione. Podział zachowuje obrys,
ale może zmieniać zestaw punktów kontaktu i wynik solvera; stare wyniki
wymagają ponownej oceny. Nie usuwa nadmiernej penetracji miękkich stóp.
Wizualizacja: `docs/calf_split_visualization_2026-10-22.html`.

W aktywnym `plane.xml` zastosowano obecnie regulator **seed 606 z 2,5 m**
i jego pozycję `home`. Kp hip/thigh/calf = 67,4 / 75,4 / 34,8;
Kd = 1,4 / 4,8 / 7,3; spine Kp/Kd = 73,5 / 5,2. Są to wartości
zaokrąglone z logu klastra. Preset: `data/landing_seed606.json`.
Poprzedni aktywny model zachowano jako `plane_before_seed606_fixed_pd.xml`.
Trening: `sbatch experiments/test2.sh` z katalogu głównego (lub
`sbatch test2.sh` z katalogu `experiments`). Oba launchery używają wspólnego
profilu `train_landing_seed606.py`.
Opis i filmy: `docs/landing_seed606_training_2026-10-22.html`.
Wzmocnienia w XML są podzielone przez gear². Reset curriculum
używa pozycji `home` aktywnego modelu jako pozycji startowej;
zerowa akcja polityki zadaje tę samą
pozycję, a kolejne akcje są odchyleniami od niej. Istniejąca randomizacja
działa wokół tych wartości. W trybie locked kąt startowy spine jest zerowany.
Wysokość zrzutu wynika z konfiguracji curriculum. Nowy profil 606 używa
`height_reference=base` w treningu i ewaluacji: 1,0–2,5 m oznacza wysokość
bazy nad podłożem. Domyślny tryb `clearance` zachowuje historyczne dodawanie
wysokości home. W pierwszej fazie treningu 606 DR fizyki/nastaw wynosi zero;
rośnie tylko wyzwanie stanu początkowego. Reguły kontaktu pozostają wspólne.

`intention.xml` wczytuje bezpośrednio
`../rl_x/environments/custom_mujoco/robot_locomotion/robots/silver_badger/data/plane.xml`.
Ten plik RL jest źródłem geometrii, bezwładności, zakresów stawów, przekładni,
limitów silników i jawnych par kontaktu. Wrapper ustawia takie same parametry
jak `LocomotionEnv`: fizyka 5 ms, solver Newton 100/50 i włączone eulerdamp.
Nie ma już osobnej kopii dynamiki robota ani podłoża z `scene_flat.xml`.
Przenosząc optymalizator na klaster, należy zachować układ katalogów projektu
wraz z XML i assets robota RL.

CPU i MJX używają actuatorów `position`: Kp/Kd kandydata są joint-side,
więc do parametrów actuatora trafiają po podzieleniu przez gear², a zadana
pozycja jest mnożona przez gear. Obowiązuje przycinanie celu do zakresu stawu
oraz ograniczanie prędkości co 20 ms, jak w RL. Nie stosujemy dodatkowej
martwej strefy momentu ze starego kontrolera ROS. Wysiłek silników jest
liczony z rzeczywistego `actuator_force`, a `--safety-factor` ogranicza
`forcerange`, nie pozycję zadaną. Wariant zablokowany odtwarza zakres,
sztywność, tłumienie i armature zablokowanego spine w środowisku RL.

Ścieżka MJX zachowuje fizyczny collider `rear_mesh`, jawne pary kontaktu
i ustawienia solvera. Pomijane są jedynie meshe wizualne bez kontaktów.

Nazwy parametrów i pliki JSON pozostają zgodne z dotychczasowymi skryptami:
`rl_j0` itd. są mapowane na nazwy stawów i actuatorów RL. Dotychczasowe
nastawy można odtworzyć, ale trzeba je ponownie ocenić w nowej fizyce.

Poprawka kontaktu łydek (2026-10-16): wspólny `plane.xml` zawiera jawne
pary podłogi z `RL_calf`, `RR_calf`, `FR_calf` i `FL_calf`. Wcześniej maski
kolizji tych cylindrów były zerowe i łydki mogły przechodzić przez podłogę,
nie generując ani siły kontaktu, ani flagi `non_foot_contact`.
Dotyczyło to także modelu treningowego. Oceny bezpieczeństwa i nastawy
uzyskane bez tych kontaktów wymagają ponownej weryfikacji.
Kontakt używa tych samych parametrów co `rear_mesh`; przy uderzeniu może
pozostawać niewielkie przejściowe zagłębienie wynikające z miękkiego kontaktu
i kroku fizyki 5 ms. Test fizyki sprawdza rzeczywisty kontakt każdej łydki,
siłę reakcji i klasyfikację kontaktu.

Historyczny limit **1000 N średniej siły w oknie 10 ms na każdą łydkę**
pozostaje pomiarem i dodatkową karą za przeciążenie górnej części. Sam kontakt górnej łydki
już dyskwalifikuje, niezależnie od siły. Przekroczenie limitu dodaje karę
`LANDING_QUALITY_PENALTY * peak_calf_force / MAX_LANDING_CALF_FORCE`.
Równe 1000 N nie przekracza limitu. To próg optymalizacji uzgodniony
z użytkownikiem, nie pomiar wytrzymałości konstrukcji. Fizyka kontaktu,
`shin_contact` i pomiar siły łydki pozostają aktywne. Kontakt innych części
robota nadal dyskwalifikuje, podobnie jak przekroczenie istniejących limitów
przyspieszenia lub siły stopy. Reguła jest wspólna dla CPU i MJX.
Raporty i CSV używają teraz nazwy `disallowed_contact` zamiast
`non_foot_contact`; obecnie górna część łydki jest niedozwolonym kontaktem.

Koszt górnej części łydki obejmuje obciążenie i czas podparcia. Normalne siły kontaktu są
sumowane osobno dla każdej łydki, a następnie uśredniane w przesuwającym się
oknie 10 ms (przed kontaktem historia zawiera zera). Do kosztu trafia:

```
calf_force_cost = integral(sum((mean_force_per_calf_10ms / robot_weight)**2)) dt
calf_support_cost = 5 * czas_kontaktu_dowolnej_lydki_w_sekundach
```

Kwadrat obciążenia mocniej zniechęca do silnego uderzenia w jedną łydkę.
Czas podparcia zlicza próbki, w których siła dowolnej łydki przekracza 1 N;
ten próg pomija drobne siły numeryczne. Wagi `CALF_FORCE_COST_WEIGHT=1`
i `CALF_SUPPORT_COST_PER_SECOND=5` są początkowymi wagami optymalizacji,
a nie limitami wytrzymałości konstrukcji. Każda sekunda podparcia dowolnymi
łydkami dodaje 5 punktów, niezależnie od ich liczby. Wyniki zawierają także
czas kontaktu, pik chwilowy i pik średniej 10 ms osobno dla RL/RR/FR/FL.
`peak_shin_force` oznacza obecnie pik sumy sił górnych części łydek.

Koszty i limit obowiązują w CPU i MJX, replay, sweep i nowych wynikach CSV.
Nie zmieniają fizyki. Surowy chwilowy pik jest raportowany oddzielnie;
limit sprawdza największą średnią 10 ms pojedynczej łydki, nie sumę.
Walidacja zlicza te porażki jako `fail_calf_force`. Zapisane wcześniej
nastawy należy ocenić ponownie, a optymalizację uruchomić ponownie,
by uwzględniała nowy koszt i limit.

```bash
# Kontrola dotychczasowego regulatora bez okna.
rlx/bin/python search/replay_target.py --height 3 --no-viewer

# Stopy zielone, dolna 1/3 niebieska, górne 2/3 pomarańczowe/czerwone.
# Domyślna pauza przy pierwszym niedozwolonym kontakcie wynosi 1 s.
rlx/bin/python search/replay_target.py --height 3 --show-colliders --speed 0.1

# Krótka próba uruchomienia wyszukiwania CPU.
rlx/bin/python search/main.py --trials 8 --initial-trials 8 --workers 1 \
    --height 1.5 --optimize-nominal-position --seed 42 --no-viewer \
    --output-dir /tmp/silver-badger-search-smoke

# Porównanie skompilowanych modeli i trajektorii z rzeczywistym środowiskiem RL.
rlx/bin/python -m unittest discover -s search -p test_physics.py -v
```

Ujednolicenie obejmuje nominalną fizykę. Wyszukiwanie zachowuje swój cel,
kryteria bezpieczeństwa i warunki resetu: wysokość oznacza bezwzględne z
bazy, a robot startuje w pozycji kandydata. Curriculum RL nadal dodaje
wysokość nominalną do wysokości zrzutu i ma własną randomizację oraz ocenę
sukcesu. Te różnice należy uwzględnić przy porównywaniu wyników.

Walidacja zmiany (2026-10-15): cztery testy zgodności przechodzą, w tym
porównanie trajektorii przez 2 s z właściwym `LocomotionEnv` w obu trybach
spine. Próba CPU z 8 kandydatami kończy się i zapisuje wykres. Zapisany
regulator z `replay_target.py` przy bazie na 3 m ma teraz pik średniego
przyspieszenia 118,61 m/s² i niedozwolony kontakt; nie przechodzi starego
kryterium bezpieczeństwa search. Poprzednia ocena tego regulatora dotyczyła
innej fizyki i wymaga ponownej optymalizacji.

MJX sprawdzono na backendzie CPU, w float64, przez 160 kroków fizyki
obejmujących impact z 3 m. Wyniki były skończone; pik średniego przyspieszenia
wyniósł 118,86 m/s², około 0,2% więcej niż w MuJoCo CPU. Siły stóp i limity
momentów były zgodne. Backend MJX może dawać różnice numeryczne w kontakcie
mesha, więc kandydatów GPU należy nadal walidować w MuJoCo CPU. Nie wykonywano
pomiaru na sprzęcie GPU ani nowego długiego wyszukiwania.

Istniejące testy RL: 33/34 przechodzą. `test_easy_replay_preserves_curriculum_level`
oczekuje na sztywno startu 1,0 m, podczas gdy zastana lokalna konfiguracja ma
1,5 m. Ta zmiana nie modyfikuje konfiguracji curriculum ani tego testu.

## Kontakty ud i wspólne limity (2026-10-20)

Wspólny model ma teraz fizyczne pary podłogi z meshami ud i obudów bioder.
Ich kontakt jest niedozwolony. Zapisany replay 2,3 m, wcześniej `Safe=True`,
po poprawce jest `Safe=False`. CPU i trening używają wspólnej mapy kontaktów,
monitora sił oraz progów; ścieżka MJX zachowuje wszystkie fizyczne meshe.
Okna sił 10 ms i przyspieszenia 50 ms są zerowane przed pierwszym kontaktem.
Dokładne kryteria i różnica między bezpieczeństwem a stabilizacją:
[landing_safety_shared.md](../docs/landing_safety_shared.md).
