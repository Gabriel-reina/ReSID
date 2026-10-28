"""Deterministic CPU Lloyd initialization for the controlled SID recipes."""
import torch

def nearest(x, centers):
    return (x.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * x @ centers.T).argmin(-1)

@torch.no_grad()
def torch_kmeans(x, k, seed, iterations=30):
    generator = torch.Generator().manual_seed(seed)
    chosen = [int(torch.randint(len(x), (1,), generator=generator))]
    distance = (x - x[chosen[0]]).square().sum(-1)
    for _ in range(1, k):
        if distance.sum() <= 1e-12:
            chosen.append(int(torch.randint(len(x), (1,), generator=generator)))
        else:
            chosen.append(int(torch.multinomial(distance, 1, generator=generator)))
        distance = torch.minimum(distance, (x - x[chosen[-1]]).square().sum(-1))
    centers = x[chosen].clone()
    for _ in range(iterations):
        codes = nearest(x, centers)
        sums = torch.zeros_like(centers).index_add_(0, codes, x)
        counts = torch.bincount(codes, minlength=k).float()
        update = sums / counts[:, None].clamp_min(1)
        centers = torch.where(counts[:, None] > 0, update, centers)
    return centers
