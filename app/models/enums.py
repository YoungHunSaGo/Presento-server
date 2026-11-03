from enum import StrEnum


class Purpose(StrEnum):
    INTERVIEW = "INTERVIEW"
    CLASS = "CLASS"
    PITCH = "PITCH"
    ETC = "ETC"


class AnalysisStatus(StrEnum):
    WAITING_UPLOAD = "WAITING_UPLOAD"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ScoreCategory(StrEnum):
    DELIVERY = "DELIVERY"
    NONVERBAL = "NONVERBAL"
    CONTENT = "CONTENT"
