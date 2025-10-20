
# 🧑🏻‍💻 Migong0311.github.io  
**Portfolio-as-a-Service | GitHub Pages 기반 개인 포트폴리오 웹사이트**

---

## 🏗️ 프로젝트 개요

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=30&pause=1200&color=EF31F7&width=435&lines=Build+once%2C+share+everywhere.)](https://git.io/typing-svg)

> 협업, 자기소개, AI 학습 여정, 그리고 프로젝트를 한 곳에서 정리하는 **정적 포트폴리오 서비스**입니다.

이 사이트는 GitHub Pages(`migong0311.github.io`)를 기반으로 제작된 개인 포트폴리오로,  
**HTML · CSS · JavaScript** 만을 사용한 정적 구조로 설계되었습니다.  
서버나 DB 없이도 콘텐츠를 지속적으로 확장할 수 있도록 구성하였으며,  
AI 학습 기록과 협업 성과, 프로젝트 사례를 직관적으로 표현하는 것을 목표로 합니다.

---

## 🧭 주요 페이지 구성

| 구분 | 설명 |
|------|------|
| **Home (`index.html`)** | 전체 개요, 협업 사례, AI 학습 타임라인, 대표 프로젝트, 연락처 |
| **About (`assets/html/about.html`)** | 자기소개 및 학력/경력, 기술 스택, GitHub/Algorithm 활동 |
| **AI Practice (`ai_practice.html`)** | AI 학습 노트 목록 및 모달 뷰어 (페이지네이션 10개 단위) |
| **Project Pages (`assets/projects/...`)** | 개별 프로젝트 상세 페이지 (예: Pangyo Coffee Legends) |

---

## 🧠 AI 학습 페이지 (`ai_practice.html`)

AI 학습 페이지는 다음과 같은 특징을 가집니다:

- **정적 Markdown 렌더링**  
  `assets/ai/` 폴더에 `.md` 파일을 추가하고, JS의 `AI_NOTES` 배열에 등록하면 자동 반영됩니다.
- **페이지네이션**  
  한 페이지당 10개씩 노트를 표시하고, 최신 날짜 순으로 정렬합니다.
- **모달 팝업 뷰어**  
  각 항목의 “보기” 버튼을 클릭하면 해당 Markdown 파일이 부드럽게 모달로 표시됩니다.
- **부가 기능**  
  상단/하단 스크롤 버튼, 다크모드 지원, 반응형 Bootstrap UI.

```js
// 예시: AI_NOTES 등록 방식
const AI_NOTES = [
  { title: '데이터 정규화 & 선형대수 해법', src: 'assets/ai/day02-ai-linear-algebra.md', date: '2025-10-01' },
  { title: 'MLP 구현 · 학습/평가 루프', src: 'assets/ai/day03-ai-mlp.md', date: '2025-10-02' },
];
```

---

## 🚀 대표 프로젝트

### **Pangyo Coffee Legends — 스마트 오피스 AIoT 시스템**

> “1년 뒤 판교 스타벅스에서 모이자!”

AI + IoT 기반의 스마트 오피스 시스템 구축 프로젝트로,
쾌적한 업무환경, 근태 분석, 회의실 예약, 실시간 알림 기능을 통합 구현했습니다.

| 분류           | 내용                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------- |
| **역할 (김미성)** | 근태 분석 · 리포트 · **GenAI 기반 분석**                                                               |
| **주요 기능**    | 센서 수집(MQTT/InfluxDB), AI 쾌적도 분석, ELK 로그 대시보드, 출입 통제/근태 분석, 회의실 예약                           |
| **기술 스택**    | Spring Boot · Python(FastAPI, XGBoost) · InfluxDB · Elasticsearch · Docker · GitHub Actions |
| **팀 구성**     | 8명 (AI, IoT, 백엔드, 프론트엔드 통합 협업)                                                              |

---

## ⚙️ 기술 스택

| 구분                     | 기술                                                        |
| ---------------------- | --------------------------------------------------------- |
| **Frontend**           | HTML, CSS, JavaScript (Vanilla), Bootstrap 5              |
| **Styling / UI**       | Bootstrap Icons, Custom Theme Variables, Scroll Animation |
| **Markdown Rendering** | [Marked.js](https://github.com/markedjs/marked)           |
| **Hosting**            | GitHub Pages                                              |
| **Version Control**    | Git / GitHub                                              |
| **Future Extension**   | Django + SQLite (CRUD 기능 확장 예정)                           |

---

## 🌙 기능 요약

✅ 다크/라이트 테마 토글
✅ 협업 카드 기반 섹션
✅ AI 학습 타임라인 / 페이지네이션 / 모달 뷰
✅ 대표 프로젝트 카드
✅ “맨 위로 / 맨 아래로” 고정 스크롤 버튼
✅ 반응형 디자인 (모바일/데스크톱 대응)

---

## 🧩 디렉토리 구조

```
Migong0311.github.io/
├── index.html
├── ai_practice.html
├── assets/
│   ├── css/style.css
│   ├── js/scripts.js
│   ├── html/about.html
│   ├── ai/                # 마크다운 학습 노트 저장소
│   ├── projects/          # 프로젝트 상세 페이지
│   └── icons/favicon.svg
└── README.md              # 현재 파일
```

---

## 🔧 수정 및 확장 방법

1. **AI 학습 노트 추가**

   * `assets/ai/` 폴더에 `.md` 파일 추가
   * `ai_practice.html` → `AI_NOTES` 배열에 항목 등록
   * GitHub에 커밋 후 배포 자동 반영

2. **새 프로젝트 추가**

   * `assets/projects/`에 `.html` 생성
   * `index.html`의 `#projects` 섹션에 카드 항목 추가

3. **About 페이지 업데이트**

   * `assets/html/about.html` 내 학력, 경험, 기술스택 섹션 수정
   * 내부 네비게이션 버튼으로 빠르게 이동 가능

---

## 🧑🏻‍💻 개발자 소개

**김미성 (Migong0311)**

* **전공:** 일본어학과 → 백엔드 개발 전향
* **소속:** SSAFY 14기 / NHN Academy AIoT 2기
* **관심 분야:** 백엔드 개발 · 시스템 아키텍처 설계 · AI 모델 분석 · 서비스 자동화
* **이메일:** [kimms000311@gmail.com](mailto:kimms000311@gmail.com)
* **GitHub:** [github.com/Migong0311](https://github.com/Migong0311)

---

## 📈 향후 계획

* [ ] Django 기반 버전 확장 (DB 연동, CRUD 관리)
* [ ] 프로젝트별 리포트 자동 생성 모듈
* [ ] AI 학습 노트 검색 / 태그 필터링 기능
* [ ] Markdown 자동 인덱싱 (GitHub API 연동)

---

## 🪄 라이선스

이 포트폴리오는 **개인 학습 및 공개 포트폴리오용**으로 제작되었습니다.
템플릿 및 소스 일부는 자유롭게 참고 가능하나,
전체 구조 및 디자인 복제 시 출처 표기를 권장합니다.

> © 2025 [Migong0311](https://migong0311.github.io) · Hosted on GitHub Pages



