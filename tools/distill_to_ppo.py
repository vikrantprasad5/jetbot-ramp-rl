"""Turn a distillation checkpoint into a PPO starting checkpoint (model C step 2).

actor  <- distilled student (camera + LiDAR network)
critic <- teacher's critic (same privileged 'critic' observations and architecture)

    ~/isaacsim/kit/python/bin/python3 tools/distill_to_ppo.py <distill model.pt> <teacher model.pt> <out.pt>
"""

import sys

import torch

student_ckpt, teacher_ckpt, out = sys.argv[1:4]
d = torch.load(student_ckpt, map_location="cpu", weights_only=False)
t = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
actor = d["student_state_dict"]
# PPO actor uses init_std 0.2 for exploration; keep the distilled mean network, reset the std to 0.2
for k in list(actor):
    if k.startswith("distribution.") and ("std" in k):  # action noise only, never obs_normalizer._std
        v = actor[k]
        actor[k] = torch.full_like(v, 0.2) if "log" not in k else torch.full_like(v, float(torch.log(torch.tensor(0.2))))
        print(f"reset {k}: {tuple(v.shape)}")
# PPO always restores its optimizer: give it a fresh Adam state (no moments) over the same number of parameters
# (actor params then critic params; obs-normalizer entries are buffers, not parameters)
n_params = sum(1 for k in list(actor) + list(t["critic_state_dict"]) if "obs_normalizer" not in k)
dummy = [torch.nn.Parameter(torch.zeros(1)) for _ in range(n_params)]
opt_state = torch.optim.Adam(dummy, lr=1.0e-4).state_dict()
torch.save({"actor_state_dict": actor, "critic_state_dict": t["critic_state_dict"], "optimizer_state_dict": opt_state,
            "iter": 0, "infos": None}, out)
print(f"optimizer params: {n_params}")
print(f"wrote {out}: actor keys {len(actor)}, critic keys {len(t['critic_state_dict'])}")
