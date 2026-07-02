from typing import List

from backend.models.survivor import Survivor, rank_by_urgency


def urgency_score(health: float, mobility: float, supplies: float) -> float:
    return (
        (100 - health) * 0.5
        + (100 - mobility) * 0.3
        + (100 - supplies) * 0.2
    )


def rank_survivors(survivors: List[dict]) -> List[dict]:
    objs = [Survivor.from_dict(s) for s in survivors]
    ranked = rank_by_urgency(objs)
    return [s.to_dict() for s in ranked]
