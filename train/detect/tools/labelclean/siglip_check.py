# SAM 3 라벨 상자를 잘라 SigLIP(so400m) 영(0)샷 분류 — 양성 문구 확률 합(pos)을 상자마다 JSONL 로.
# 사용: siglip_check.py <sam3.jsonl> <클래스 번호> <out.jsonl> [표본 수]  (teacher/.venv-ref 의 python 으로)
import sys, json, random, torch, numpy as np
from PIL import Image
from transformers import SiglipModel, PreTrainedTokenizerFast
W = '/home/juyoung/limo_fire_ws/train/detect/data/weights/siglip-so400m-patch14-384'
PROMPTS = {
 3: (["an emergency exit sign", "a green exit sign with a running man", "an illuminated EXIT sign", "an evacuation direction sign with an arrow"],
     ["a shop sign", "a street name sign", "a road traffic sign", "a warning sign", "a notice with text", "a fire extinguisher sign",
      "a fire hydrant sign", "a company logo", "a room number plate", "a restroom sign", "a neon sign", "a license plate"]),
 0: (["fire", "flames", "a candle flame", "a burning fire", "a wildfire", "a lighter flame", "a gas stove flame"],
     ["a lamp", "a light bulb", "a sunset", "a street light", "a company logo", "an orange object", "a car headlight", "a reflection of light", "a neon sign", "a cartoon drawing"]),
 5: (["a door", "an open door", "a glass door", "a double door", "a front door of a house", "a metal door"],
     ["a fire hydrant cabinet", "a kitchen cabinet", "a wardrobe", "a car door", "a car", "a window", "a wall", "a refrigerator", "a locker", "a picture frame"]),
}
jl, cls, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else -1
dev = 'cuda'
model = SiglipModel.from_pretrained(W, torch_dtype=torch.float16).to(dev).eval()
tok = PreTrainedTokenizerFast(tokenizer_file=W + "/tokenizer.json", pad_token="</s>", eos_token="</s>", unk_token="<unk>"); tok.model_max_length = 64
pos, neg = PROMPTS[cls]
t = tok([s.lower() for s in pos + neg], padding='max_length', max_length=64, return_tensors='pt').to(dev)
with torch.no_grad():
    tf = model.get_text_features(input_ids=t["input_ids"]); tf = getattr(tf, "pooler_output", tf); tf = tf / tf.norm(dim=-1, keepdim=True)
scale, bias = model.logit_scale.exp().float(), model.logit_bias.float()
items = []
for line in open(jl):
    r = json.loads(line)
    for k, d in enumerate(r.get('det', [])):
        if d['c'] == cls and d['s'] > 0.5: items.append((r['img'], k, d['b'], d['s']))
if limit > 0:
    random.seed(3); random.shuffle(items); items = items[:limit]
mean = torch.tensor([0.5] * 3, device=dev).view(1, 3, 1, 1)
fo = open(out, 'w'); B = 64
for i in range(0, len(items), B):
    batch = items[i:i + B]; px = []
    for p, k, b, s in batch:
        im = Image.open(p).convert('RGB'); Wd, Hd = im.size; x1, y1, x2, y2 = b; m = 0.1 * max(x2 - x1, y2 - y1)
        cr = im.crop((max(0, x1 - m), max(0, y1 - m), min(Wd, x2 + m), min(Hd, y2 + m))).resize((384, 384), Image.BICUBIC)
        px.append(torch.from_numpy(np.asarray(cr).copy()).permute(2, 0, 1))
    x = (torch.stack(px).to(dev).half() / 255 - 0.5) / 0.5
    with torch.no_grad():
        f = model.get_image_features(pixel_values=x); f = getattr(f, "pooler_output", f); f = f / f.norm(dim=-1, keepdim=True)
        lg = (f.float() @ tf.float().T) * scale
        pr = lg.softmax(-1)
    pp = pr[:, :len(pos)].sum(-1).tolist(); top = pr.argmax(-1).tolist()
    for (p, k, b, s), q, tp in zip(batch, pp, top):
        fo.write(json.dumps({'img': p, 'k': k, 'b': b, 's': s, 'pos': round(q, 4), 'top': (pos + neg)[tp]}) + '\n')
print(len(items))
