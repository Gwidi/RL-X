"""Interactive chapter selection for suspended policy-range visualization."""
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def build(table):
    path=ROOT/'docs/artifacts/policy_joint_ranges/ranges.json'
    if not path.exists() or not (ROOT/'docs/report/media/policy_joint_ranges.mp4').exists():return ''
    data=json.loads(path.read_text());rows=[];buttons=[]
    for r in data['joints']:
        deg=lambda key:f'{math.degrees(r[key]):+.1f}°'
        rows.append([r['joint'],deg('home_rad'),deg('min_rad')+' … '+deg('max_rad'),deg('mechanical_min_rad')+' … '+deg('mechanical_max_rad'),'dolny' if r['clipped_low'] else 'górny' if r['clipped_high'] else 'brak'])
        buttons.append(f'<button type="button" onclick="const v=document.getElementById(\'joint-range-video\');v.currentTime={r["start_seconds"]};v.play();" style="background:#18212a;color:#84d4c0;border:1px solid #34404b;padding:9px 12px;border-radius:4px;cursor:pointer">{r["joint"]}</button>')
    section=f'''<section id="zakres-stawow" class="stage"><div class="stage-label">Wizualizacja przestrzeni akcji</div><h2>Robot zawieszony: pełny zakres pozycji zadanych polityki</h2>
<p>Film pokazuje kolejno wszystkie 13 stawów. Każdy wykonuje przebieg home → akcja −1 → akcja +1 → home; pozostałe pozostają w home. Korpus bazowy jest unieruchomiony w powietrzu. Dwa widoki z przeciwnych stron pomagają zobaczyć również dalsze kończyny. Żółty punkt na robocie oznacza aktywny staw; zielony odcinek pod obrazem to zakres dostępny polityce, szary to limity modelu, a biała kreska to home.</p>
<video id="joint-range-video" controls preload="metadata" poster="media/policy_joint_ranges.png" style="width:100%;height:auto"><source src="media/policy_joint_ranges.mp4" type="video/mp4"></video>
<p>Wybierz staw, aby przejść do jego fragmentu (6 sekund na staw):</p><div style="display:flex;flex-wrap:wrap;gap:8px">{''.join(buttons)}</div>
<p><a href="media/policy_joint_ranges.mp4" download>Pobierz film MP4</a> · <a href="../artifacts/policy_joint_ranges/ranges.json">Dokładne zakresy w radianach (JSON)</a></p>
<h3>Co dokładnie oglądamy?</h3><p>Pozycje wyliczono przez tę samą funkcję PDControl.process_action, której używa środowisko: home + skala × akcja, następnie ograniczenie do limitów stawu i uwzględnienie przełożenia. Skala wynosi 0,6 rad, czyli około 34,4° w każdą stronę przed obcięciem. Offsety resetu są zerowe. Są to zakresy współrzędnych stawów, a nie kąty przechyłu całego robota.</p>
<p>To wizualizacja kinematyczna: bez całkowania dynamiki, swobodnego spadku i wywoływania polityki best/latest. Ustawienie stawu na granicy pokazuje dopuszczalną komendę, ale nie dowodzi, że napęd osiągnie ją w dostępnym czasie lub że wszystkie kombinacje pozycji są wolne od samokolizji. Podczas ruchu kręgosłupa tylna część korpusu obraca się względem unieruchomionej bazy.</p>
{table(['Staw','Home','Zakres polityki','Limity modelu','Obcięty koniec zakresu'],rows)}
<p>RL/RR: lewa/prawa tylna noga; FL/FR: lewa/prawa przednia noga. Hip oznacza staw biodrowy, thigh staw uda, calf staw łydki; spine to kręgosłup. Znaki kątów wynikają z osi modelu i różnią się między stronami robota.</p>
<p><strong>Wniosek:</strong> zwiększenie std nie rozszerzy pokazanych zielonych odcinków. Ich rozszerzenie wymaga zmiany mapowania akcji lub pozycji odniesienia i nadal podlega limitom stawów. Obcięcie występuje dla RL_thigh, RR_thigh i FL_calf. Film pozwala wskazać, które ruchy warto dalej zbadać dynamicznie.</p></section>'''
    (ROOT/'docs/report/joint-ranges.html').write_text('''<!doctype html><html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zakres ruchu polityki PPO</title><style>body{margin:0;background:#11171e;color:#dce4ec;font:17px/1.7 system-ui}main{max-width:1400px;margin:auto;padding:28px}h1,h2,h3{line-height:1.3}a{color:#84d4c0}.stage-label{color:#84d4c0}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #34404b;white-space:nowrap}button:hover{filter:brightness(1.3)}@media(max-width:600px){main{padding:16px}}</style><main>'''+section+'</main></html>')
    return section
