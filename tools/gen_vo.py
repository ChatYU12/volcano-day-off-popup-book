import re, json, hashlib, os, subprocess, sys, numpy as np, soundfile as sf, warnings, random
warnings.filterwarnings('ignore')
from scipy.signal import fftconvolve, butter, sosfilt, resample_poly
sys.path.insert(0,'/home/claude/tts')
from kk import synth
SRC=sys.argv[1]; OUT=sys.argv[2]
src=open(SRC).read()
lines=set()
for m in re.finditer(r"R\.say\((pupu|bird|peek),'([^']*)'(,\{[^}]*\})?\)",src):
    o=m.group(3) or ''
    if 'sfx' in o: continue
    who=m.group(1); role='bd' if who in('bird','peek') else ('ppq' if 'soft' in o else 'pp')
    lines.add((role,m.group(2)))
for m in re.finditer(r"R\.narr\('([^']*)'",src): lines.add(('n',m.group(1)))
pk=re.search(r"const POKE=\{(.*?)\};",src).group(1)
for who,arr in re.findall(r"(pp|bd):\[(.*?)\]",pk):
    for t in re.findall(r"'([^']*)'",arr): lines.add((who,t))
lines=sorted(lines);print(len(lines),'lines');[print(' ',l) for l in lines]
CAST={'n':('zf_xiaoxiao',.92,1.0,1.0,.07,None),'pp':('zm_yunxi',.95,.97,1.0,.05,None),'ppq':('zm_yunxi',.86,.97,.75,.09,5500),'bd':('zf_xiaoyi',1.05,1.12,1.0,.05,None)}
SR=24000;rng=np.random.default_rng(7)
ir=rng.standard_normal(int(SR*.4))*np.exp(-np.linspace(0,7,int(SR*.4)));ir/=np.abs(ir).sum()/6
def clean(t):
    t=t.replace('「','').replace('」','').replace('——','').replace('……','').replace('—','')
    return t.strip('，') or t
os.makedirs(OUT,exist_ok=True);man={}
for r,t in lines:
    v,sp,pf,g,wet,lp=CAST[r];random.seed(hash((r,t))%9999);jit=1+random.uniform(-.03,.03)
    a,_=synth(clean(t),v,speed=sp*jit/pf)
    if abs(pf-1)>1e-3: a=resample_poly(a,int(round(1000/pf)),1000)
    if lp: a=sosfilt(butter(4,lp,'low',fs=SR,output='sos'),a)
    a=a+fftconvolve(a,ir)[:len(a)]*wet*4
    nz=np.where(np.abs(a)>0.01)[0]
    if len(nz): a=a[max(0,nz[0]-480):nz[-1]+1200]
    a=a*(0.075*g/(np.sqrt(np.mean(a**2))+1e-9));a=np.clip(a,-.97,.97)
    a=np.concatenate([np.zeros(int(SR*.06)),a,np.zeros(int(SR*.12))])
    key=f'{r}|{t}';f=f"{r}_{hashlib.md5(key.encode()).hexdigest()[:10]}.mp3"
    sf.write('/tmp/x.wav',a,SR);subprocess.run(['ffmpeg','-y','-loglevel','error','-i','/tmp/x.wav','-ac','1','-b:a','48k',f'{OUT}/{f}'],check=True)
    man[key]={'f':f,'d':round(len(a)/SR,2)}
json.dump(man,open(f'{OUT}/vo.json','w'),ensure_ascii=False,indent=0)
print('total',round(sum(m['d'] for m in man.values()),1),'s')
