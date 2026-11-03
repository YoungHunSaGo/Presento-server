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


class AnalysisStage(StrEnum):
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING_NONVERBAL = "ANALYZING_NONVERBAL"
    ANALYZING_CONTENT = "ANALYZING_CONTENT"
    GENERATING_FEEDBACK = "GENERATING_FEEDBACK"


# 명세 6장: 단계별 progress 범위 (시작, 끝)
STAGE_PROGRESS: dict[AnalysisStage, tuple[int, int]] = {
    AnalysisStage.EXTRACTING_AUDIO: (0, 10),
    AnalysisStage.TRANSCRIBING: (10, 40),
    AnalysisStage.ANALYZING_NONVERBAL: (40, 65),
    AnalysisStage.ANALYZING_CONTENT: (65, 80),
    AnalysisStage.GENERATING_FEEDBACK: (80, 100),
}


class VideoSource(StrEnum):
    UPLOAD = "UPLOAD"
    RECORDING = "RECORDING"


class ScoreCategory(StrEnum):
    DELIVERY = "DELIVERY"
    NONVERBAL = "NONVERBAL"
    CONTENT = "CONTENT"
