import torch
from torch import tensor

from geomstats.geometry.torus import Torus

if __name__ == "__main__":
    torus = Torus(2)

    v1 = torch.tensor([[1.0], [2.0]])
    v2 = torch.tensor([[3.0], [4.0]])

    euclidean_distance = torch.norm(v1 - v2)
    geodesic_distance = torus.metric.dist(v1, v2)

    print(f"Euclidean distance: {euclidean_distance}")
    print(f"Geodesic distance: {geodesic_distance}")
