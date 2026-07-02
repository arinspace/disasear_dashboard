from typing import Optional, List


class Survivor:
    def __init__(self, survivor_id: str, lat: float, lng: float,
                 health: float = 80, mobility: float = 80, supplies: float = 60):
        self.id = survivor_id
        self.lat = lat
        self.lng = lng
        self.health = health
        self.mobility = mobility
        self.supplies = supplies

    def urgency(self) -> float:
        return (
            (100 - self.health) * 0.5
            + (100 - self.mobility) * 0.3
            + (100 - self.supplies) * 0.2
        )

    def to_dict(self):
        return {
            "id": self.id,
            "lat": self.lat,
            "lng": self.lng,
            "health": self.health,
            "mobility": self.mobility,
            "supplies": self.supplies,
            "urgency": round(self.urgency(), 1),
        }

    @staticmethod
    def from_dict(d):
        return Survivor(
            survivor_id=d.get("id", "Unknown"),
            lat=d["lat"],
            lng=d["lng"],
            health=d.get("health", 80),
            mobility=d.get("mobility", 80),
            supplies=d.get("supplies", 60),
        )


def survivors_from_list(data):
    return [Survivor.from_dict(s) for s in data]


def rank_by_urgency(survivors: List[Survivor]) -> List[Survivor]:
    return sorted(survivors, key=lambda s: s.urgency(), reverse=True)
