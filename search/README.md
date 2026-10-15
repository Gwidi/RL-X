Optymalizacja korzysta z nominalnej fizyki treningu Silver Badger.

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

```bash
# Kontrola dotychczasowego regulatora bez okna.
rlx/bin/python search/replay_target.py --height 3 --no-viewer

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
