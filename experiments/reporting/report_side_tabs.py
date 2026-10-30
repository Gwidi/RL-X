"""Side navigation and accessible stage panels for the local research report."""
import html,re,json

CSS='''
body{padding-left:264px}html{scroll-padding-top:24px}.masthead{padding:32px 0 26px}.masthead h1{font-size:30px;max-width:none;margin:10px 0 12px}.masthead .intro{font-size:16px;margin-bottom:12px}.project-progress{margin-top:18px;padding-top:14px;gap:8px 24px}.stage{padding:36px 0 56px}.stage-label{margin-bottom:16px}h2{max-width:32ch}.side-nav{position:fixed;inset:0 auto 0 0;width:264px;height:100dvh;z-index:30;background:#0c1218;border-right:1px solid #293641;overflow-y:auto;padding:28px 18px 24px}.side-brand{display:block;font-size:18px;font-weight:650;color:#f1f5fa;text-decoration:none;padding:0 12px 22px;line-height:1.4}.side-brand small{display:block;color:#8d9dab;font-size:11px;letter-spacing:.12em;text-transform:uppercase;margin-top:8px}.side-label{font-size:11px;letter-spacing:.12em;color:#788a99;text-transform:uppercase;padding:20px 12px 8px}.side-links{display:flex;flex-direction:column;gap:4px}.side-links a{display:flex;gap:12px;align-items:flex-start;padding:12px;border-radius:5px;color:#a3b1bf;text-decoration:none;font-size:14px;line-height:1.45;border-left:2px solid transparent}.side-links a:hover{background:#15212b;color:white}.side-links a[aria-selected=true]{color:#b3eddf;background:#172e32;border-left-color:#84d4c0}.tab-index{color:#68818e;min-width:20px;font-variant-numeric:tabular-nums}.side-links a[aria-selected=true] .tab-index{color:#84d4c0}.side-links a small{display:block;font-size:11px;color:#7f939f;margin-top:3px}.side-toggle{display:none}.tab-panel{outline:none}.tab-panel[hidden]{display:none!important}.stage-pager{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;padding-top:32px;margin-top:44px;border-top:1px solid #34404b;font-size:14px}.stage-pager a{text-decoration:none}.stage-pager a:last-child{margin-left:auto}.films{grid-template-columns:minmax(0,1fr)}.films video{max-height:720px;object-fit:contain}.nav-shade{display:none}.skip{z-index:100}a:focus-visible,button:focus-visible{outline:2px solid #84d4c0;outline-offset:4px}
@media(max-width:1050px){body{padding-left:230px}.side-nav{width:230px;padding-inline:12px}.container{width:calc(100% - 48px)}}
@media(max-width:720px){body{padding-left:58px}.side-nav{width:58px;padding:18px 5px;overflow-x:hidden}.side-toggle{display:block;background:transparent;color:#dce4ec;border:1px solid #34404b;border-radius:5px;width:38px;height:38px;margin:0 auto 18px;cursor:pointer;font-size:22px}.side-brand{font-size:0;padding:0;height:10px}.side-brand small,.side-label,.tab-text{display:none}.side-links{gap:7px}.side-links a{padding:11px 7px;font-size:12px;gap:0;justify-content:center}.tab-index{min-width:0}.side-links a[aria-selected=true]{border-left-width:2px}.container{width:calc(100% - 32px)}.masthead{padding-top:24px}.masthead h1{font-size:24px;line-height:1.3}.masthead .intro{display:none}.project-progress{font-size:12px;gap:6px}.stage{padding-top:26px}h2{font-size:28px}h3{font-size:23px}.nav-open .side-nav{width:264px;padding:18px 14px}.nav-open .side-brand{font-size:18px;height:auto;padding:0 12px 20px}.nav-open .side-brand small,.nav-open .side-label,.nav-open .tab-text{display:block}.nav-open .side-links a{justify-content:flex-start;gap:12px;font-size:14px;padding:12px}.nav-open .side-toggle{margin-left:auto;margin-right:0}.nav-open .nav-shade{display:block;position:fixed;inset:0;background:#0008;z-index:29;border:0}.films{gap:24px}}
@media print{body{padding-left:0}.side-nav,.nav-shade,.stage-pager{display:none!important}.tab-panel[hidden]{display:block!important}.masthead{padding-top:0}.container{width:100%}}
'''

JS='''
(function(){
 const panels=[...document.querySelectorAll('main.container > section')];
 const tabs=[...document.querySelectorAll('.side-links [role="tab"]')];
 panels.forEach(p=>{p.classList.add('tab-panel');p.setAttribute('role','tabpanel');p.setAttribute('tabindex','-1');p.setAttribute('aria-labelledby','tab-'+p.id)});
 const toggle=document.querySelector('.side-toggle');
 function closeMenu(){document.body.classList.remove('nav-open');toggle.setAttribute('aria-expanded','false')}
 toggle.addEventListener('click',()=>{const open=document.body.classList.toggle('nav-open');toggle.setAttribute('aria-expanded',String(open))});
 document.querySelector('.nav-shade').addEventListener('click',closeMenu);
 function activate(hash,scroll=true){
  let id;try{id=decodeURIComponent((hash||'').replace(/^#/,''))}catch(e){id='aktualnie'}
  let target=document.getElementById(id)||document.getElementById('aktualnie');
  let panel=target.closest('.tab-panel')||document.getElementById('aktualnie');
  panels.forEach(p=>{p.hidden=p!==panel;if(p.hidden)p.querySelectorAll('video').forEach(v=>v.pause())});
  tabs.forEach(a=>{const active=a.hash==='#'+panel.id;a.setAttribute('aria-selected',String(active));a.tabIndex=active?0:-1;if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')});
  for(let n=target;n&&n!==panel;n=n.parentElement)if(n.tagName==='DETAILS')n.open=true;
  closeMenu();
  if(scroll)requestAnimationFrame(()=>{if(target===panel)window.scrollTo({top:0,behavior:'instant'});else target.scrollIntoView({behavior:'instant',block:'start'})});
 }
 tabs.forEach((a,i)=>a.addEventListener('keydown',e=>{let j;if(e.key==='ArrowDown')j=(i+1)%tabs.length;else if(e.key==='ArrowUp')j=(i-1+tabs.length)%tabs.length;else if(e.key==='Home')j=0;else if(e.key==='End')j=tabs.length-1;else return;e.preventDefault();tabs[j].focus();history.pushState(null,'',tabs[j].hash);activate(tabs[j].hash)}));
 document.addEventListener('click',e=>{const a=e.target.closest('a[href^="#"]');if(!a)return;const hash=a.getAttribute('href');if(!document.getElementById(hash.slice(1)))return;e.preventDefault();if(location.hash!==hash)history.pushState(null,'',hash);activate(hash)});
 window.addEventListener('hashchange',()=>activate(location.hash));window.addEventListener('popstate',()=>activate(location.hash));
 panels.forEach((p,i)=>{const nav=document.createElement('nav');nav.className='stage-pager';nav.setAttribute('aria-label','Przejścia między etapami');for(const j of [i-1,i+1]){if(j<0||j>=panels.length)continue;const a=document.createElement('a');a.href='#'+panels[j].id;const tab=tabs.find(t=>t.hash===a.hash);a.textContent=(j<i?'← ':'')+(tab?tab.textContent.trim().replace(/\\s+/g,' '):panels[j].id)+(j>i?' →':'');nav.append(a)}p.append(nav)});
 activate(location.hash||'#aktualnie',false);
})();
'''

def apply_side_tabs(result,data):
    links=[('aktualnie','•','Aktualny stan','Podsumowanie i następny krok')]
    for i,s in enumerate(data['stages'],1):links.append((f'e{i}',f'{i:02d}',s['title'],s['date']))
    for branch in data.get('research_branches',[]):
        links.append((branch['id'],'B1',branch['title'],branch['status']))
    links.append(('dokumenty','↗','Dokumenty i dane','Materiały źródłowe'))
    anchors=''.join(f'<a href="#{id}" id="tab-{id}" role="tab" aria-controls="{id}" aria-selected="false" aria-label="{html.escape(title)}" title="{html.escape(title)}"><span class="tab-index">{num}</span><span class="tab-text">{html.escape(title)}<small>{html.escape(note)}</small></span></a>' for id,num,title,note in links)
    anchors=anchors.replace('<a href="#branch-aerial"','<div class="side-label">Równoległy tor badawczy</div><a href="#branch-aerial"')
    aside=f'''<aside class="side-nav" aria-label="Etapy raportu"><button class="side-toggle" type="button" aria-expanded="false" aria-controls="stage-tabs" aria-label="Rozwiń lub zwiń etapy">☰</button><a class="side-brand" href="#aktualnie">Emergency landing<small>Silver Badger / PPO</small></a><div class="side-label">Raport badań · kolejne etapy</div><nav id="stage-tabs" class="side-links" role="tablist" aria-orientation="vertical" aria-label="Zakładki etapów">{anchors}</nav><div class="side-label">Przejdź do szczegółów</div><div class="side-links"><a href="#faktoryzacja" title="Wyjaśnienie faktoryzacji"><span class="tab-index">ƒ</span><span class="tab-text">Faktoryzacja</span></a><a href="#zakres-stawow" title="Film: zakres ruchu stawów"><span class="tab-index">▶</span><span class="tab-text">Zakres stawów · film</span></a></div></aside><button class="nav-shade" aria-label="Zamknij menu etapów"></button>'''
    result=re.sub(r'<nav class="navigation".*?</nav>',lambda _:aside,result,count=1,flags=re.S)
    result=re.sub(r'<script>.*?</script>',lambda _:'<script>'+JS+'</script>',result,flags=re.S)
    result=result.replace('</style>',CSS+'</style>',1)
    return result
