from typing import List, Optional


class Route:
    def __init__(self, path_m: List[dict], cost: float, algo_name: str, extra: str = ""):
        self.path_m = path_m
        self.cost = cost
        self.algo_name = algo_name
        self.extra = extra

    def to_dict(self):
        return {
            "path": self.path_m,
            "cost": round(self.cost, 4),
            "algo_name": self.algo_name,
            "extra": self.extra,
        }
