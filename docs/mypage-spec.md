# Presento 마이페이지 명세 (화면 기획 + API M1–M3)

> **AI 코딩 도구에게:** 이 문서는 Presento 마이페이지를 만들기 위한 기준 명세다. 기존 `docs/api-spec.md`(공통 규칙, 에러 형식, 인증)를 그대로 따른다.
> - 경로, 필드명(camelCase), enum 값은 이 문서에 적힌 그대로 쓴다. 없는 필드를 지어내지 않는다.
> - 백엔드 연결 전에는 5장의 예시 JSON을 MSW 목업으로 그대로 응답한다.
> - 모호한 부분은 임의로 정하지 말고 `TODO(spec):` 주석을 남기고 사람에게 묻는다.
> - 차트는 기존 권장 스택의 Recharts를 쓴다.

---

## 1. 목적

연습 데이터가 쌓이면 사용자가 **"나는 어디가 강하고 어디가 약한지, 실력이 늘고 있는지"**를 한 화면에서 보게 한다.

- 숫자(평균, 추이, 강점·약점 판정)는 전부 서버 코드가 계산한다. AI 추가 호출은 없다(비용 0).
- 집계 대상은 **분석 완료(`COMPLETED`)된 회차만**이다. 실패한 회차, 삭제된 발표·회차는 빠진다.
- 날짜 기준은 한국 시간(Asia/Seoul)이다.

## 2. 화면 기획 `/mypage`

상단 내비게이션에 "마이페이지" 링크를 추가한다. 로그인 필요(401이면 로그인 화면).

### 레이아웃 (위에서 아래 순서)

| 영역 | 내용 | 데이터 |
| --- | --- | --- |
| 프로필 헤더 | 이름, 이메일, 가입일 | A3 `GET /users/me` |
| 필터 바 | 기간 토글 `30일 / 90일 / 전체`(기본 전체), 목적 드롭다운 `전체 / 면접 / 수업 / 피칭 / 기타` | 세 API 모두 같은 `period`, `purpose`로 다시 호출 |
| 요약 카드 4개 | 완료한 연습 수, 총 연습 시간, 평균 점수(+ 최근 변화 화살표), 연속 연습일 | M1 |
| 점수 추이 차트 | 꺾은선: x=회차 날짜, y=종합 점수(0~100). 범례로 전달력·비언어·내용 선 토글 | M3 |
| 강점 / 약점 카드 | 좌: 강점 최대 3개, 우: 약점 최대 3개. 카드마다 항목명, 점수, 추이 화살표, 수치 문구, 한 줄 조언 | M2 |
| 다음 연습 집중 포인트 | `focusTip` 한 문장을 강조 박스로 | M2 |
| 항목별 점수 | 레이더 차트(카테고리 3개 평균) + 세부 항목 표(점수 막대, 추이) | M1 `categoryAverages`, M2 `metrics` |
| 자주 쓰는 추임새 | 상위 5개 단어와 횟수(막대 또는 태그) | M2 `topFillers` |
| 자주 받은 피드백 | 피드백 유형 상위 5개 | M2 `frequentFeedbackTypes` |
| 주간 활동 | 막대: 주별 연습 횟수(최근 12주, 0인 주 포함) | M1 `weeklyActivity` |
| 목적별 기록 | 목적별 연습 수·평균 점수 표 | M1 `purposeBreakdown` |

### 동작 규칙

- **데이터가 적을 때:** M2 `status`가 `INSUFFICIENT_DATA`면 강점·약점·집중 포인트 영역 대신 "분석을 {requiredCount}회 이상 완료하면 강점과 약점을 알려드려요. (현재 {analysisCount}회)"를 보여준다. 요약 카드와 차트는 데이터가 있는 만큼 그린다.
- **완료한 연습이 0회:** 전체를 빈 상태로 바꾸고 "첫 발표를 연습해 보세요" 버튼(발표 만들기로 이동)을 보여준다.
- **null 값:** `averageScore`, `recentChange`, `bestScore` 등은 데이터가 없으면 `null`이다. 화면에는 `-`로 표시한다.
- **추이 화살표:** `trend`가 `UP`이면 초록 위 화살표, `DOWN`이면 빨강 아래 화살표, `FLAT`이면 회색 가로선. `UP`은 항상 "좋아짐"이다(추임새가 줄어도 `UP`).
- **점수 추이 차트 점 클릭:** 해당 리포트 `/analyses/{analysisId}`로 이동.
- **강점·약점이 비어 있음:** 강점이 없으면 "아직 뚜렷한 강점이 없어요", 약점이 없으면 "뚜렷한 약점이 없어요. 좋아요!"를 보여준다.
- 기간·목적을 바꾸면 세 API를 함께 다시 부른다. TanStack Query 키 예: `['mypage', 'stats', period, purpose]`.

## 3. API 목록 (기존 16개 + 3개)

| # | 분류 | Method | Path | 설명 | 인증 |
| --- | --- | --- | --- | --- | --- |
| M1 | 마이페이지 | GET | `/users/me/stats` | 요약 통계, 카테고리 평균, 주간 활동, 목적별 기록 | O |
| M2 | 마이페이지 | GET | `/users/me/insights` | 강점·약점, 항목별 점수, 추임새·피드백 순위 | O |
| M3 | 마이페이지 | GET | `/users/me/score-timeline` | 발표 구분 없이 회차별 점수 추이 | O |

### 공통 쿼리 파라미터

| 이름 | 값 | 기본 | 설명 |
| --- | --- | --- | --- |
| `period` | `30d`, `90d`, `all` | `all` | 회차 완료 시각 기준, 오늘 포함 최근 30일·90일 |
| `purpose` | `INTERVIEW`, `CLASS`, `PITCH`, `ETC` | 없음(전체) | 해당 목적의 발표만 |

잘못된 값이면 400 `VALIDATION_ERROR`. 토큰이 없거나 만료면 401 `UNAUTHORIZED`. 다른 사용자 데이터는 구조상 볼 수 없다(항상 내 데이터만).

## 4. API 상세

### M1. 요약 통계 `GET /users/me/stats?period=all&purpose=`

응답 `200`

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `period` | string | 요청한 기간 값 |
| `from` | string \| null | 집계 시작 시각(ISO 8601). `all`이면 `null` |
| `to` | string | 집계 기준 시각(지금) |
| `presentationCount` | number | 완료 회차가 1개 이상 있는 발표 수 |
| `analysisCount` | number | 완료 회차 수 |
| `totalPracticeSec` | number | 완료 회차 영상 길이 합(초, 소수점 1자리) |
| `averageScore` | number \| null | 종합 점수 평균(정수, 반올림) |
| `bestScore` | number \| null | 최고 종합 점수 |
| `bestAnalysisId` | string \| null | 최고 점수 회차 ID |
| `recentChange` | number \| null | 최근 5회 평균 − 그 직전 5회 평균(정수). 완료 회차가 6회 미만이면 `null` |
| `streakDays` | number | 오늘 또는 어제까지 하루도 빠짐없이 연습한 날 수. 어제도 오늘도 안 했으면 0 |
| `categoryAverages` | object | `DELIVERY`, `NONVERBAL`, `CONTENT` 평균(정수). 측정된 회차가 없으면 그 키 값은 `null` |
| `gradeDistribution` | object | 등급별 회차 수. 키는 `A+`, `A`, `B+`, `B`, `C+`, `C`, `D` 전부 항상 포함 |
| `purposeBreakdown` | array | `{ purpose, analysisCount, averageScore }`. 회차가 있는 목적만, 회차 수 많은 순 |
| `weeklyActivity` | array | 최근 12주(월요일 시작), 오래된 순, 0인 주 포함. `{ weekStart(YYYY-MM-DD), analysisCount, practiceSec, averageScore(null 가능) }` |

```json
{
  "period": "all",
  "from": null,
  "to": "2026-11-05T20:00:00+09:00",
  "presentationCount": 4,
  "analysisCount": 12,
  "totalPracticeSec": 2874.5,
  "averageScore": 74,
  "bestScore": 88,
  "bestAnalysisId": "a-9d21",
  "recentChange": 6,
  "streakDays": 3,
  "categoryAverages": { "DELIVERY": 70, "NONVERBAL": 76, "CONTENT": 79 },
  "gradeDistribution": { "A+": 0, "A": 1, "B+": 3, "B": 4, "C+": 2, "C": 1, "D": 1 },
  "purposeBreakdown": [
    { "purpose": "CLASS", "analysisCount": 7, "averageScore": 75 },
    { "purpose": "INTERVIEW", "analysisCount": 5, "averageScore": 72 }
  ],
  "weeklyActivity": [
    { "weekStart": "2026-08-17", "analysisCount": 0, "practiceSec": 0, "averageScore": null },
    { "weekStart": "2026-10-26", "analysisCount": 5, "practiceSec": 1210.3, "averageScore": 72 },
    { "weekStart": "2026-11-02", "analysisCount": 7, "practiceSec": 1664.2, "averageScore": 76 }
  ]
}
```
(예시는 길이를 줄이려고 주 일부를 생략했다. 실제 응답은 항상 12개다.)

### M2. 강점·약점 `GET /users/me/insights?period=all&purpose=`

응답 `200`

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `status` | `READY` \| `INSUFFICIENT_DATA` | 완료 회차가 `requiredCount` 미만이면 `INSUFFICIENT_DATA` |
| `analysisCount` | number | 기간·목적 조건의 완료 회차 수 |
| `requiredCount` | number | 강점·약점을 내는 최소 회차 수(현재 3) |
| `basedOnRecent` | number | 판정에 쓴 최근 회차 수(최대 10) |
| `strengths` | Insight[] | 최대 3개, 점수 높은 순. `INSUFFICIENT_DATA`면 빈 배열 |
| `weaknesses` | Insight[] | 최대 3개, 점수 낮은 순. `INSUFFICIENT_DATA`면 빈 배열 |
| `metrics` | Insight[] | 측정된 모든 항목, 아래 `metric` 표 순서 |
| `focusTip` | string \| null | 가장 약한 항목 기준 다음 연습 조언 한 문장. 약점이 없거나 데이터 부족이면 `null` |
| `topFillers` | array | `{ word, count }` 최대 5개, 많은 순 |
| `frequentFeedbackTypes` | array | `{ type, label, count }` 최대 5개, 많은 순. `type`은 기존 `feedbacks.type` 값 |

**Insight 객체**

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `metric` | string | 아래 표의 값 |
| `category` | string | `DELIVERY`, `NONVERBAL`, `CONTENT` |
| `label` | string | 화면 표시 이름 |
| `score` | number | 최근 회차들의 항목 점수 평균(0~100 정수) |
| `trend` | `UP` \| `DOWN` \| `FLAT` | 판정 구간 안에서 최근 절반 평균 − 이전 절반 평균. ±5점 미만이면 `FLAT`. 표본 4회 미만이면 `FLAT` |
| `change` | number \| null | 위 차이(정수). 표본 4회 미만이면 `null` |
| `sampleCount` | number | 이 항목이 측정된 회차 수 |
| `valueText` | string | 실제 수치를 사람이 읽는 문구(예: "분당 3.4회") |
| `message` | string | 한 줄 설명·조언(해요체) |

**`metric` 값**

| metric | category | label | valueText 예 |
| --- | --- | --- | --- |
| `PACE` | DELIVERY | 말 속도 | 분당 341음절 |
| `FILLER` | DELIVERY | 추임새 | 분당 3.4회 |
| `SILENCE` | DELIVERY | 긴 침묵 | 분당 0.8회 |
| `TIME` | DELIVERY | 시간 관리 | 제한 시간의 92% 사용 |
| `GAZE` | NONVERBAL | 시선 | 정면 78% |
| `POSTURE` | NONVERBAL | 자세 | 안정성 86% |
| `GESTURE` | NONVERBAL | 제스처 | 분당 6.2회 |
| `STRUCTURE` | CONTENT | 구조 | 회차당 지적 1.2건 |
| `CLARITY` | CONTENT | 명확성 | 회차당 지적 0.8건 |
| `LANGUAGE` | CONTENT | 언어 표현 | 회차당 부적절 표현 0.3건 |

판정 규칙(서버): 강점은 `score >= 80`, 약점은 `score < 70`, 둘 다 `sampleCount >= 2`인 항목만. 측정되지 않은 회차(예: 손이 안 보여 제스처 미측정, 제한 시간 없음)는 그 항목 평균에서 빠진다.

```json
{
  "status": "READY",
  "analysisCount": 12,
  "requiredCount": 3,
  "basedOnRecent": 10,
  "strengths": [
    { "metric": "GAZE", "category": "NONVERBAL", "label": "시선", "score": 89, "trend": "UP", "change": 8, "sampleCount": 10, "valueText": "정면 79%", "message": "청중을 안정적으로 바라보고 있어요." }
  ],
  "weaknesses": [
    { "metric": "FILLER", "category": "DELIVERY", "label": "추임새", "score": 58, "trend": "UP", "change": 12, "sampleCount": 10, "valueText": "분당 4.8회", "message": "'약간', '이제'가 자주 나와요. 문장 사이에 말 대신 잠깐 쉬어 보세요." },
    { "metric": "PACE", "category": "DELIVERY", "label": "말 속도", "score": 63, "trend": "FLAT", "change": 2, "sampleCount": 10, "valueText": "분당 424음절", "message": "말이 조금 빨라요. 핵심 문장 앞에서 속도를 늦춰 보세요." }
  ],
  "metrics": [
    { "metric": "PACE", "category": "DELIVERY", "label": "말 속도", "score": 63, "trend": "FLAT", "change": 2, "sampleCount": 10, "valueText": "분당 424음절", "message": "말이 조금 빨라요. 핵심 문장 앞에서 속도를 늦춰 보세요." },
    { "metric": "FILLER", "category": "DELIVERY", "label": "추임새", "score": 58, "trend": "UP", "change": 12, "sampleCount": 10, "valueText": "분당 4.8회", "message": "'약간', '이제'가 자주 나와요. 문장 사이에 말 대신 잠깐 쉬어 보세요." },
    { "metric": "GAZE", "category": "NONVERBAL", "label": "시선", "score": 89, "trend": "UP", "change": 8, "sampleCount": 10, "valueText": "정면 79%", "message": "청중을 안정적으로 바라보고 있어요." }
  ],
  "focusTip": "다음 연습에서는 추임새를 분당 1회 이하로 줄이는 데 집중해 보세요.",
  "topFillers": [ { "word": "약간", "count": 21 }, { "word": "이제", "count": 14 }, { "word": "음", "count": 9 } ],
  "frequentFeedbackTypes": [ { "type": "FILLER", "label": "추임새", "count": 9 }, { "type": "PACE_FAST", "label": "말 속도 빠름", "count": 6 } ]
}
```

데이터 부족 예:

```json
{ "status": "INSUFFICIENT_DATA", "analysisCount": 2, "requiredCount": 3, "basedOnRecent": 2, "strengths": [], "weaknesses": [], "metrics": [], "focusTip": null, "topFillers": [ { "word": "음", "count": 3 } ], "frequentFeedbackTypes": [] }
```

### M3. 점수 추이 `GET /users/me/score-timeline?period=all&purpose=&limit=50`

- `limit`: 1~200, 기본 50. 조건에 맞는 완료 회차 중 **최근 `limit`개**를 **오래된 순**으로 준다.
- 발표 하나의 추이는 기존 P7을 쓰고, 마이페이지는 발표 구분 없는 전체 추이에 M3을 쓴다.

응답 `200`

```json
{
  "items": [
    { "analysisId": "a-1c4f", "presentationId": "p-7f3c", "presentationTitle": "캡스톤 중간 발표", "purpose": "CLASS", "attemptNo": 1, "createdAt": "2026-11-03T21:00:00+09:00", "overallScore": 64, "grade": "C+", "scores": { "DELIVERY": 58, "NONVERBAL": 70, "CONTENT": 65 } },
    { "analysisId": "a-9d21", "presentationId": "p-7f3c", "presentationTitle": "캡스톤 중간 발표", "purpose": "CLASS", "attemptNo": 3, "createdAt": "2026-11-05T20:00:00+09:00", "overallScore": 82, "grade": "B+", "scores": { "DELIVERY": 80, "NONVERBAL": 84, "CONTENT": 81 } }
  ],
  "total": 12
}
```

`total`은 조건에 맞는 전체 완료 회차 수(잘리기 전)다.

## 5. 프론트 타입 (TypeScript)

```ts
export type Period = '30d' | '90d' | 'all';
export type Purpose = 'INTERVIEW' | 'CLASS' | 'PITCH' | 'ETC';
export type Category = 'DELIVERY' | 'NONVERBAL' | 'CONTENT';
export type Trend = 'UP' | 'DOWN' | 'FLAT';
export type Metric =
  | 'PACE' | 'FILLER' | 'SILENCE' | 'TIME'
  | 'GAZE' | 'POSTURE' | 'GESTURE'
  | 'STRUCTURE' | 'CLARITY' | 'LANGUAGE';
export type Grade = 'A+' | 'A' | 'B+' | 'B' | 'C+' | 'C' | 'D';

export interface MyStats {
  period: Period;
  from: string | null;
  to: string;
  presentationCount: number;
  analysisCount: number;
  totalPracticeSec: number;
  averageScore: number | null;
  bestScore: number | null;
  bestAnalysisId: string | null;
  recentChange: number | null;
  streakDays: number;
  categoryAverages: Record<Category, number | null>;
  gradeDistribution: Record<Grade, number>;
  purposeBreakdown: { purpose: Purpose; analysisCount: number; averageScore: number | null }[];
  weeklyActivity: { weekStart: string; analysisCount: number; practiceSec: number; averageScore: number | null }[];
}

export interface Insight {
  metric: Metric;
  category: Category;
  label: string;
  score: number;
  trend: Trend;
  change: number | null;
  sampleCount: number;
  valueText: string;
  message: string;
}

export interface MyInsights {
  status: 'READY' | 'INSUFFICIENT_DATA';
  analysisCount: number;
  requiredCount: number;
  basedOnRecent: number;
  strengths: Insight[];
  weaknesses: Insight[];
  metrics: Insight[];
  focusTip: string | null;
  topFillers: { word: string; count: number }[];
  frequentFeedbackTypes: { type: string; label: string; count: number }[];
}

export interface ScoreTimelineItem {
  analysisId: string;
  presentationId: string;
  presentationTitle: string;
  purpose: Purpose;
  attemptNo: number;
  createdAt: string;
  overallScore: number;
  grade: Grade;
  scores: Record<Category, number>;
}

export interface ScoreTimeline {
  items: ScoreTimelineItem[];
  total: number;
}
```

## 6. 프론트 구현 순서

1. MSW 목업 3개(M1–M3)를 4장 예시 그대로 추가한다. `INSUFFICIENT_DATA`와 완료 0회 빈 상태도 목업으로 확인할 수 있게 쿼리 파라미터나 개발용 토글로 전환 가능하게 만든다.
2. `/mypage` 라우트와 내비게이션 링크를 추가한다.
3. 요약 카드 → 점수 추이 차트 → 강점·약점 카드 → 나머지 순서로 만든다. 강점·약점 카드가 시연의 핵심이다.
4. 기간·목적 필터를 연결한다.
5. 실제 서버에 붙여 확인한다. 연습 데이터가 적으면 같은 발표를 몇 번 더 분석해 3회 이상 쌓는다.

## 7. 아직 정해지지 않은 것

- [ ] 강점·약점 기준값(80 / 70)과 최소 회차 수(3)는 평가셋으로 조정할 수 있다. 값이 바뀌어도 응답 모양은 같다.
- [ ] 목표 점수 설정, 배지 같은 동기 부여 기능은 이번 범위가 아니다.
