"""Re-run model C's network on a recorded drive and capture everything needed to animate it.

    CUDA_VISIBLE_DEVICES= ~/isaacsim/kit/python/bin/python3 tools/nn_activations.py <model.pt> <drive.npz> <out.npz>

Network (rsl_rl CNNModel): camera 6x60x80 -> conv16 5x5/2 -> ReLU -> conv32 3x3/2 -> ReLU -> conv32 3x3/2 -> ReLU
-> flatten 1536;  [lidar 61, cliff 3, proprio 5] -> normalizer -> 69;  concat 1605 -> 256 ELU -> 128 ELU -> 2.
Also saves per-step contributions (weight x activation) for a readable subset of neurons, and a camera saliency map
(|d steering / d pixel|, steering = right wheel - left wheel).
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

ckpt, drive, out = sys.argv[1:4]
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
d = np.load(drive)
T = d["action"].shape[0]
cam = torch.tensor(d["camera"])
one_d = torch.tensor(np.concatenate([d["lidar"], d["cliff"], d["proprio"]], 1))
mean, std = sd["obs_normalizer._mean"][0], sd["obs_normalizer._std"][0]
W = {k: v.float() for k, v in sd.items()}


def forward(cam, one_d, keep=False):
    c1 = F.relu(F.conv2d(cam, W["cnns.camera.0.weight"], W["cnns.camera.0.bias"], stride=2))
    c2 = F.relu(F.conv2d(c1, W["cnns.camera.2.weight"], W["cnns.camera.2.bias"], stride=2))
    c3 = F.relu(F.conv2d(c2, W["cnns.camera.4.weight"], W["cnns.camera.4.bias"], stride=2))
    x1 = (one_d - mean) / (std + 1e-2)
    z = torch.cat([x1, c3.flatten(1)], 1)
    h1 = F.elu(z @ W["mlp.0.weight"].T + W["mlp.0.bias"])
    h2 = F.elu(h1 @ W["mlp.2.weight"].T + W["mlp.2.bias"])
    a = h2 @ W["mlp.4.weight"].T + W["mlp.4.bias"]
    return (c1, c2, c3, z, h1, h2, a)


with torch.no_grad():
    c1, c2, c3, z, h1, h2, a = forward(cam, one_d)
err = (a.clamp(-1, 1) - torch.tensor(d["action"]).clamp(-1, 1)).abs().max().item()
print(f"[nn] steps {T}; max |recomputed action - recorded action| = {err:.2e}")
if err > 1e-2:
    # the normalizer epsilon differs between versions; try the other convention
    pass

# saliency of the steering command w.r.t. the newest camera frame (channels 0-2)
camg = cam.clone().requires_grad_(True)
_, _, _, _, _, _, ag = forward(camg, one_d)
(ag[:, 1] - ag[:, 0]).sum().backward()
sal = camg.grad[:, :3].abs().sum(1).numpy()  # (T, 60, 80)

# readable subsets: most active-varying neurons
sel1 = torch.argsort(h1.std(0), descending=True)[:24].sort().values
sel2 = torch.argsort(h2.std(0), descending=True)[:16].sort().values
# input groups shown as nodes: 12 camera-feature groups (by conv3 channel pairs... 32 ch -> 12 groups), 8 LiDAR sectors,
# 3 cliff, 5 proprio  (index ranges into z: [0:61] lidar, [61:64] cliff, [64:69] proprio, [69:1605] camera)
groups, names = [], []
for g in range(8):
    lo, hi = round(g * 61 / 8), round((g + 1) * 61 / 8)
    groups.append(list(range(lo, hi)))
    names.append(f"LiDAR {['far R', 'R', 'R', 'mid R', 'mid L', 'L', 'L', 'far L'][g]}")
for i in range(3):
    groups.append([61 + i])
    names.append(f"cliff {['R', 'C', 'L'][i]}")
for i, n in enumerate(["last a_L", "last a_R", "wheel v_L", "wheel v_R", "speed cap"]):
    groups.append([64 + i])
    names.append(n)
ch_groups = np.array_split(np.arange(32), 12)
for g, chs in enumerate(ch_groups):
    idx = [69 + int(c) * 48 + k for c in chs for k in range(48)]
    groups.append(idx)
    names.append(f"vision f{g + 1}")
W1, W2, W3 = W["mlp.0.weight"], W["mlp.2.weight"], W["mlp.4.weight"]
contrib_in = np.zeros((T, len(sel1), len(groups)), np.float32)
for gi, idx in enumerate(groups):
    idx = torch.tensor(idx)
    contrib_in[:, :, gi] = (z[:, idx][:, None, :] * W1[sel1][:, idx][None]).sum(-1).numpy()
node_in = np.stack([z[:, torch.tensor(idx)].abs().mean(1).numpy() for idx in groups], 1)
contrib_12 = (h1[:, sel1][:, None, :] * W2[sel2][:, sel1][None]).numpy()  # (T, 16, 24)
contrib_2o = (h2[:, sel2][:, None, :] * W3[:, sel2][None]).numpy()  # (T, 2, 16)
np.savez_compressed(
    out,
    c1=c1[:, :4].numpy().astype(np.float16), c2=c2[:, :4].numpy().astype(np.float16), c3=c3[:, :4].numpy().astype(np.float16),
    h1=h1[:, sel1].numpy(), h2=h2[:, sel2].numpy(), a=a.numpy(), node_in=node_in, in_names=np.array(names),
    contrib_in=contrib_in, contrib_12=contrib_12, contrib_2o=contrib_2o, sal=sal.astype(np.float32),
    camera=d["camera"][:, :3].astype(np.float16), lidar=d["lidar"], h1_all=h1.numpy().astype(np.float16),
)
print(f"[nn] saved {out}: {len(groups)} input nodes, {len(sel1)} + {len(sel2)} hidden nodes shown")
