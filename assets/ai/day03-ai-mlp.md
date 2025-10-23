# 🧠 Day 03 — 머신러닝 기초 실습 (시험 대비용 종합 정리)

---

## 📍 1. 데이터 로딩 및 탐색적 데이터 분석 (EDA)

### ✅ 1️⃣ 데이터 불러오기

```python
from sklearn.datasets import load_wine
import pandas as pd

df, y = load_wine(as_frame=True, return_X_y=True)
df["quality"] = y
```

| 항목     | 설명                       |
| -------- | -------------------------- |
| 데이터셋 | Wine (sklearn 내장 데이터) |
| 샘플 수  | 178개                      |
| 특성 수  | 13개 feature + 1개 target  |
| 클래스   | 3개 (0, 1, 2)              |

> 🔹 시험포인트: `load_wine(as_frame=True)` → `pandas.DataFrame` 반환
> 🔹 `return_X_y=True` → `(X, y)` 형태로 직접 받음

---

### ✅ 2️⃣ 데이터 요약 및 통계 확인

```python
df.info()
df.describe()
df["quality"].value_counts()
```

**주요 함수 설명**

* `.info()` → 열(column) 수, 결측치 여부, dtype 확인
* `.describe()` → 기본 통계량(평균, 표준편차, 사분위수)
* `.value_counts()` → 클래스 불균형 확인

> 🔹 시험포인트: 데이터 분포 확인 목적 → EDA(탐색적 데이터 분석) 단계

---

### ✅ 3️⃣ 간단한 통계 예시

```python
(df["color_intensity"] >= 10).mean() * 100
df.groupby("quality")["alcohol"].mean()
```

> 🔹 비율 계산 시 `(조건).mean()` 사용
> 🔹 `groupby`로 클래스별 평균·분포 차이 확인 가능

---

## 📊 2. 시각화를 통한 데이터 탐색

### ✅ 1️⃣ 상관관계 Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(14, 12))
sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("하삼각 상관계수 히트맵")
plt.show()
```

**핵심 개념**

* `.corr()` : 수치형 변수 간의 선형 상관계수 (-1~1)
* `np.triu()` : 상삼각 부분 마스킹 → 중복 제거
* `annot=True` : 수치값 표시

> 🔹 시험포인트: “상관계수 해석” — **0.7 이상이면 강한 양의 상관**

---

### ✅ 2️⃣ 분포 및 산점도 시각화

```python
sns.histplot(data=df, x="flavanoids", hue="quality", kde=True)
sns.scatterplot(data=df, x="flavanoids", y="total_phenols", hue="quality")
```

| 차트          | 목적                     |
| ------------- | ------------------------ |
| `histplot`    | 단일 변수의 분포 시각화  |
| `hue` 인자    | 클래스별 분포 비교       |
| `scatterplot` | 두 변수 간 상관관계 확인 |

> 🔹 시험포인트: Seaborn의 `hue`는 그룹 구분용 색상 변수
> 🔹 `kde=True` → 밀도추정선 표시

---

## ⚙️ 3. 데이터 전처리 (Preprocessing)

### ✅ 1️⃣ Train/Test 분할

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
```

| 옵션           | 설명                                     |
| -------------- | ---------------------------------------- |
| `test_size`    | 테스트 데이터 비율                       |
| `stratify=y`   | 클래스 비율을 train/test에 동일하게 유지 |
| `random_state` | 재현성 확보                              |

> 🔹 시험포인트: `stratify` 미설정 시 **클래스 불균형 문제 발생 가능**

---

### ✅ 2️⃣ 표준화 (Standardization)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
```

| 함수               | 역할                      |
| ------------------ | ------------------------- |
| `.fit()`           | 평균(μ), 표준편차(σ) 계산 |
| `.transform()`     | Z-score 정규화 수행       |
| `.fit_transform()` | 한 번에 수행              |

> 🔹 시험포인트: **훈련 데이터로만 fit**, 테스트엔 transform만 수행해야 데이터 누수 방지

---

## 🤖 4. 지도학습 — 로지스틱 회귀

### ✅ 1️⃣ 모델 학습

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_norm, y_train)
```

| 옵션           | 설명                          |
| -------------- | ----------------------------- |
| `max_iter`     | 수렴 보장 (기본 100에서 늘림) |
| `random_state` | 재현성                        |
| `fit()`        | 모델 학습                     |
| `predict()`    | 클래스 예측                   |

> 🔹 시험포인트: **로지스틱 회귀는 선형모델 + 시그모이드(이진)**
> 🔹 `ConvergenceWarning` → `max_iter` 부족 시 발생

---

### ✅ 2️⃣ 평가 지표

```python
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

| 지표      | 의미                          |
| --------- | ----------------------------- |
| Precision | 예측 중 실제 정답 비율        |
| Recall    | 실제 정답 중 맞춘 비율        |
| F1-score  | Precision & Recall의 조화평균 |

> 🔹 시험포인트: **Precision과 Recall trade-off 관계**
> 🔹 `classification_report`는 F1-score 자동 포함

---

### ✅ 3️⃣ ROC-AUC 평가

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_score = clf.predict_proba(X_test_norm)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
auc = roc_auc_score(y_test, y_score)
```

> 🔹 ROC 곡선: **TPR(민감도) vs FPR(오탐율)**
> 🔹 AUC = 1이면 완벽, 0.5는 랜덤 추측 수준
> 🔹 시험에 자주 출제: “ROC-AUC는 모델의 **분류 성능(일반화력)** 을 나타냄”

---

## 🔁 5. 교차 검증 (Cross Validation)

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

f1_scores = cross_val_score(pipe, X, y, cv=5, scoring="f1")
print("Average F1-score:", f1_scores.mean())
```

| 항목              | 설명                                    |
| ----------------- | --------------------------------------- |
| `Pipeline`        | 여러 단계(전처리 + 모델)를 한 번에 학습 |
| `cross_val_score` | 데이터 분할 편향 줄임                   |
| `cv=5`            | 5-Fold 교차 검증                        |
| `scoring="f1"`    | 평가 척도 지정 가능                     |

> 🔹 시험포인트: 교차검증은 “**데이터 편향 최소화, 모델 일반화 성능 평가**”
> 🔹 자주 출제: KFold의 기본 원리

---

## 🌀 6. 비지도학습 — PCA + K-Means

### ✅ 1️⃣ PCA (차원 축소)

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
```

| 항목           | 설명                                   |
| -------------- | -------------------------------------- |
| `PCA`          | 데이터의 분산을 최대화하는 축으로 투영 |
| `n_components` | 축의 개수 (차원 수)                    |
| `.fit()`       | 고유벡터(주성분) 계산                  |
| `.transform()` | 데이터 변환                            |

> 🔹 시험포인트: “PCA는 비지도 차원축소 기법이며, 회전(직교 변환) 기반이다.”
> 🔹 주성분 1, 2의 분산 비율 합이 높을수록 정보 손실 적음

---

### ✅ 2️⃣ K-Means 군집화

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
```

| 항목             | 설명                                |
| ---------------- | ----------------------------------- |
| `n_clusters`     | 군집 개수                           |
| `.fit_predict()` | 학습 + 예측 동시 수행               |
| `n_init`         | 중심 초기화 반복 횟수 (성능 안정화) |

> 🔹 시험포인트: “K-Means는 중심점 거리 최소화(유클리드 거리) 기반 알고리즘”
> 🔹 랜덤 초기화 문제를 `n_init`으로 완화

---

### ✅ 3️⃣ 시각화 및 해석

```python
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y)
```

* **왼쪽:** 모델이 만든 군집 (K-Means)
* **오른쪽:** 실제 라벨 분포
  → 완벽히 일치하지 않아도 **데이터의 구조적 분리 가능성** 파악 가능

> 🔹 시험포인트: “비지도학습은 라벨 없이 구조를 파악하는 방법”
> 🔹 PCA+KMeans 조합은 시각적으로 군집 경향을 파악하는 대표적 조합

---

## 📘 7. 종합 요약 (시험 대비표)

| 구분           | 핵심 내용                      | 주요 함수                            |
| -------------- | ------------------------------ | ------------------------------------ |
| **EDA**        | 데이터 구조·통계·상관관계 파악 | `info`, `describe`, `corr`           |
| **시각화**     | 변수 분포·관계 파악            | `sns.histplot`, `sns.scatterplot`    |
| **전처리**     | 표준화, 분할, 누수 방지        | `train_test_split`, `StandardScaler` |
| **지도학습**   | 분류/예측                      | `LogisticRegression`, `predict`      |
| **평가**       | 성능 측정                      | `confusion_matrix`, `ROC-AUC`        |
| **교차검증**   | 일반화 성능 확보               | `cross_val_score`, `Pipeline`        |
| **비지도학습** | 구조 발견·차원축소             | `PCA`, `KMeans`                      |

---

## 💯 시험에 자주 나오는 개념 요약

| 키워드                      | 출제 포인트                  |
| --------------------------- | ---------------------------- |
| **표준화(Standardization)** | 평균=0, 분산=1로 스케일 조정 |
| **정규화(Normalization)**   | [0,1] 범위로 스케일 조정     |
| **Overfitting**             | 훈련 정확도↑, 테스트 정확도↓ |
| **Confusion Matrix**        | TN, FP, FN, TP 구조          |
| **ROC-AUC**                 | 분류기의 종합적 성능 평가    |
| **Cross Validation**        | 데이터 분할 편향 최소화      |
| **PCA 주성분**              | 분산이 가장 큰 방향          |
| **KMeans 수렴 기준**        | 중심점 이동이 거의 없을 때   |
| **Pipeline 장점**           | 전처리-학습 일관성 유지      |
| **ConvergenceWarning**      | `max_iter` 부족 시 발생      |

---

✅ **한줄정리:**

> “Day 03 실습은 EDA → 전처리 → 지도학습(로지스틱 회귀) → 평가 → 교차검증 → 비지도학습(PCA+KMeans)”
> 의 전체 흐름을 scikit-learn 기반으로 구현하는 **머신러닝 기본기 통합 실습**이다.


