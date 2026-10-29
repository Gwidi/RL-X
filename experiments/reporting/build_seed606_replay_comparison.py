"""Build the consolidated three-controller report from saved evaluations."""
import json
import html
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs'
OUT=DOC/'artifacts/seed606_replay_comparison_2026-10-24'

def main():
    meta=json.loads((OUT/'checkpoints.json').read_text())
    held=json.loads((OUT/'heldout_results.json').read_text())
    dirs={n:DOC/f'artifacts/seed606_replay_{n}_2026-10-24' for n in ['best','latest']}
    nominal={n:json.loads((d/'policy_results.json').read_text()) for n,d in dirs.items()}
    pd=json.loads((dirs['best']/'pd_results.json').read_text())
    def link(p):return str(p.relative_to(DOC))
    def video(d,r):
        return f'<video controls preload="none" poster="{link(d/(r["slug"]+".png"))}"><source src="{link(d/(r["slug"]+".mp4"))}" type="video/mp4"></video><p>{"Sukces" if r["success"] else "Porażka: "+html.escape(r["reason"])} · {r["end_time_s"]:.2f} s · stopa {r["peak_foot_force_10ms_N"]:.0f} N · przyspieszenie {r["peak_body_acceleration_50ms_m_s2"]:.1f} m/s²</p>'
    rows='';sections=''
    for h in [1.,1.5,2.,2.5]:
        cells=[]
        for name in ['zero','best','latest']:
            subset=[r for r in held if r['controller']==name and r['height']==h]
            assert len(subset)==32
            count=sum(r['success'] for r in subset)
            cells.append(f'{count}/32 ({count/32:.1%})')
        rows+=f'<tr><td>{h:g} m</td>'+''.join(f'<td>{c}</td>' for c in cells)+'</tr>'
        sources=[('Zerowa akcja · PD606',dirs['best'],next(r for r in pd['trials'] if r['height_m']==h))]
        sources += [(n,dirs[n],next(r for r in nominal[n]['trials'] if r['height_m']==h)) for n in ['best','latest']]
        sections+=f'<section><h2>Nominalnie: {h:g} m wysokości bazy</h2><div class="three">'+''.join(f'<div><h3>{n}</h3>{video(d,r)}</div>' for n,d,r in sources)+'</div></section>'
    totals={n:sum(r['success'] for r in held if r['controller']==n) for n in ['zero','best','latest']}
    latest25=next(r for r in nominal['latest']['trials'] if r['height_m']==2.5)
    text='''<!doctype html><html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PPO replay — best, latest i zerowa akcja</title><style>body{margin:0;background:#f2f5f7;color:#193346;font:16px/1.65 system-ui}main{max-width:1650px;margin:auto;padding:28px}h1{font-size:38px;line-height:1.2}section{background:white;padding:26px;margin:24px 0;border:1px solid #dbe5ec;border-radius:10px}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}video{width:100%;background:#142c3c;border-radius:8px}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}.note{border-left:4px solid #347e9f;padding:18px;background:#edf4f8}code{overflow-wrap:anywhere}a{color:#226e9d}@media(max-width:1050px){.three{grid-template-columns:1fr}main{padding:15px}}</style><main><h1>Drugi trening PPO: best, latest i zerowa akcja</h1><p>Trening 9295070 · nominalny replay · 24 października 2026</p><section><h2>Co porównujemy?</h2><p>We wszystkich trzech wariantach używamy identycznego regulatora PD606, home, ograniczeń momentów, geometrii, miękkiego kontaktu i kroku fizyki. Nie zmieniamy nastaw między wysokościami ani checkpointami. Zerowa akcja oznacza utrzymywanie home przez PD. PPO dodaje 13 korekt zadanej pozycji stawów; nie dobiera Kp ani Kd.</p><p>Kp nóg (hip/thigh/calf): 67,4 / 75,4 / 34,8; Kd: 1,4 / 4,8 / 7,3; spine: Kp 73,5, Kd 5,2. Akcja PPO ma skalę 0,6 rad i clipping do ograniczeń stawów. Fizyka 5 ms, sterowanie 50 Hz. Górne 2/3 łydki są zabronione, dolna 1/3 pozostaje dozwolona.</p><p>Best: 23 068 672 kroki. Latest: 30 015 488 kroków. Zapisane w latest metryki ewaluacyjne dotyczą 29 360 128 kroków, dlatego faktyczny latest sprawdzono ponownie lokalnie.</p></section><section><h2>Nowy test z zaburzeniami</h2><p>32 sparowane scenariusze na wysokość i kontroler, nowy seed 20261024. Roll i pitch niezależnie ±3°, vx i vy niezależnie ±3 m/s. Brak losowania fizyki, nowych nastaw i popchnięć w locie. Wszystkie początkowe hashe modeli i stanów są zgodne między trzema kontrolerami. Akcje polityki są deterministyczne.</p><table><tr><th>Wysokość</th><th>Zerowa akcja</th><th>Best</th><th>Latest</th></tr>'''+rows+'''</table><p>Łącznie: '''+', '.join(f'{n}: {c}/128 ({c/128:.1%})' for n,c in totals.items())+'''. To mała próba kontrolna, nie dowód istotności statystycznej ani generalizacji na inne seedy treningu.</p></section><section><h2>Wnioski</h2><p>Regulator samodzielnie rozwiązuje cztery nominalne lądowania. Korzyści z PPO należy zatem szukać w odporności na przechył i prędkość poziomą, a nie w samym osiągnięciu wysokości 2,5 m. Nowe wagi best zachowują nominalny wynik bazowego PD.</p><p>Faktyczny latest na 2,5 m: '''+('sukces' if latest25['success'] else 'porażka: '+html.escape(latest25['reason']))+'''. Dalszy trening po najlepszym checkpointcie nie daje gwarancji poprawy; należy wybierać model na podstawie niezależnej ewaluacji oraz wymagać zachowania nominalnych lądowań.</p><p>W ewaluacji klastrowej best osiągnął 35,94% przy zaburzeniach i 100% nominalnie, podczas gdy zero-action PD osiągał 3,52% przy zaburzeniach i 100% nominalnie. Jest to zestaw używany do selekcji checkpointu; powyższy test używa nowych stanów początkowych. Nie należy utożsamiać statystyk ostatniej ewaluacji z rzeczywistym wynikiem wag latest.</p><p>Nominalny replay częściowo rozwiązał problem zapominania: pojawił się checkpoint łączący nominalne sukcesy z odpornością. Nie usunął niestabilności dalszego treningu. Kolejnym krokiem jest zachowanie best jako kandydata i powtórzenie eksperymentu na kilku seedach z rozdzielonym raportowaniem nominalnej skuteczności i odporności.</p><div class="note">Filmy pokazują pojedyncze nominalne próby. Powtarzanie identycznego stanu 64 razy nie jest 64 niezależnymi scenariuszami. Miękki kontakt i penetracja stóp pozostają ograniczeniem modelu; niczego nie dostrajano pod te filmy.</div></section>'''+sections+'''<section><h2>Pochodzenie i dane</h2><p><a href="artifacts/seed606_replay_comparison_2026-10-24/checkpoints.json">Metadane i hashe checkpointów</a> · <a href="artifacts/seed606_replay_comparison_2026-10-24/heldout_results.json">Wszystkie nowe próby</a> · <a href="landing_seed606_replay_best_2026-10-24.html">Szczegóły i protokół best</a> · <a href="landing_seed606_replay_latest_2026-10-24.html">Szczegóły i protokół latest</a></p></section></main></html>'''
    note = ('<section><h2>Najważniejszy wynik</h2><p><strong>Best, latest i zerowa akcja '
            'przechodzą wszystkie cztery nominalne lądowania, w tym 2,5 m.</strong> '
            'Końcowy latest odzyskał nominalny sukces mimo porażek w ostatniej '
            'wcześniejszej ewaluacji. To koryguje wniosek oparty wyłącznie na logu.</p>'
            '<p>W nowym zestawie zaburzeń best ma 34/128 sukcesów, latest 33/128, '
            'a zerowa akcja 4/128. Różnica best–latest to jeden epizod; ten test '
            'nie daje podstaw do uznania jednego checkpointu za wyraźnie lepszy. '
            'Przewaga obu polityk nad niezmienionym PD dotyczy odporności na zaburzenia.</p></section>')
    text=text.replace('<section><h2>Co porównujemy?</h2>',note+'<section><h2>Co porównujemy?</h2>')
    (DOC/'landing_seed606_replay_comparison_2026-10-24.html').write_text(text)
    print(totals)

if __name__=='__main__':main()
