from typing import Optional


class Hazard:
    def __init__(self, lat: float, lng: float, severity: float = 5,
                 radius_m: float = 500, confidence: float = 0,
                 hazard_type: str = "flood", level_label: str = "Unknown"):
        self.lat = lat
        self.lng = lng
        self.severity = severity
        self.radius_m = radius_m
        self.confidence = confidence
        self.type = hazard_type
        self.level_label = level_label

    def to_dict(self):
        return {
            "lat": self.lat,
            "lng": self.lng,
            "severity": self.severity,
            "radius_m": self.radius_m,
            "confidence": self.confidence,
            "type": self.type,
            "level_label": self.level_label,
        }

    @staticmethod
    def from_dict(d):
        return Hazard(
            lat=d["lat"],
            lng=d["lng"],
            severity=d.get("severity", 5),
            radius_m=d.get("radius_m", 500),
            confidence=d.get("confidence", 0),
            hazard_type=d.get("type", "flood"),
            level_label=d.get("level_label", "Unknown"),
        )


def hazards_from_list(data):
    return [Hazard.from_dict(h) for h in data]
