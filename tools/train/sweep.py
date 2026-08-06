"""Find segmenter settings that feed the model what it was trained on."""
import numpy as np, torch
import simulate_app as sim
from train import SignHead
from vocabulary_features import SIGNATURE_LENGTH, vocabulary_signature

b = np.load("data/test_raw.npz", allow_pickle=True); n = int(b["n"][0])
concepts = sorted(set(np.load("data/train_raw.npz", allow_pickle=True)["y"]))
idx = {c: i for i, c in enumerate(concepts)}
model = SignHead(len(concepts))
w = np.fromfile("../../public/models/lse-vocabulary.bin", dtype=np.float32)
st = model.state_dict(); off = 0
for k in st:
    s = st[k].numel(); st[k] = torch.tensor(w[off:off+s]).reshape(st[k].shape); off += s
model.load_state_dict(st); model.eval()
y = torch.tensor([idx[str(b["y"][i])] for i in range(n)])

def evaluate(thr, drop, pad):
    sim.MOTION_THRESHOLD, sim.DECELERATION_DROP = thr, drop
    mats, dropped = [], 0
    for i in range(n):
        r, l, p, f = b[f"r{i}"], b[f"l{i}"], b[f"p{i}"], b[f"f{i}"]
        picks = sim.segment(r, l)
        if picks is None:
            dropped += 1; mats.append(np.zeros(SIGNATURE_LENGTH, np.float32)); continue
        lo = max(0, picks[0] - pad); hi = min(len(r), picks[-1] + 1 + pad)
        k = np.arange(lo, hi)
        mats.append(vocabulary_signature(r[k], l[k], p[k], f[k]))
    with torch.no_grad():
        t3 = model(torch.tensor(np.stack(mats))).topk(3, 1).indices
    return (t3[:,0]==y).float().mean().item(), (t3==y.unsqueeze(1)).any(1).float().mean().item(), dropped

print(f"{'umbral':>7} {'caida':>7} {'pad':>4}  top1   top3   perdidos")
for thr in (0.08, 0.05, 0.03, 0.02):
    for drop in (0.45, 0.35):
        for pad in (0, 6):
            t1, t3, d = evaluate(thr, drop, pad)
            print(f"{thr:>7} {drop:>7} {pad:>4}  {t1:.3f}  {t3:.3f}   {d}")
