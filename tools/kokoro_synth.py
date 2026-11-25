import zipfile, json, re, numpy as np, onnxruntime as ort, importlib.resources as ir, warnings
warnings.filterwarnings('ignore')
from misaki import zh
VOCAB=json.load(open(ir.files('kokoro_onnx')/'config.json'))['vocab']
G2P=zh.ZHG2P()
SESS=ort.InferenceSession('/home/claude/tts/kokoro-v1.0-q8.onnx',providers=['CPUExecutionProvider'])
INAMES=[i.name for i in SESS.get_inputs()]
def voice(name):
    z=zipfile.ZipFile(f'/home/claude/tts/voices/{name}.pt')
    n=[f for f in z.namelist() if re.search(r'/data/0$',f)][0]
    return np.frombuffer(z.read(n),dtype=np.float32).reshape(-1,1,256)
def synth(text,vname,speed=1.0):
    ph,_=G2P(text)
    ids=[VOCAB[c] for c in ph if c in VOCAB][:510]
    v=voice(vname) if isinstance(vname,str) else vname
    style=v[len(ids)]
    feed={INAMES[0]:np.array([[0,*ids,0]],dtype=np.int64),INAMES[1]:style.astype(np.float32),INAMES[2]:np.array([speed],dtype=np.float32)}
    return SESS.run(None,feed)[0].reshape(-1),ph
if __name__=='__main__':
    print(INAMES,[i.type for i in SESS.get_inputs()])
    import soundfile as sf
    for v in ['zf_xiaoxiao','zm_yunxia','zf_xiaoyi']:
        a,ph=synth('搬到山上的第一天，小象豆豆从海边带来了一枚贝壳。',v,0.95);sf.write(f'/tmp/t_{v}.wav',a,24000);print(v,len(a)/24000,ph)
