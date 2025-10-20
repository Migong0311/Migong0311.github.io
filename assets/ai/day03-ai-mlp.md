
# 🧠 머신러닝 기초 실습 정리 (with scikit-learn)

## 📍 1. 데이터 로딩 및 탐색적 데이터 분석 (EDA)

### ✅ 데이터 불러오기

```python
from sklearn.datasets import load_wine
import pandas as pd

df, y = load_wine(as_frame=True, return_X_y=True)
df["quality"] = y
```

### ✅ 데이터 개요

| 항목                     | 내용                        |
| ---------------------- | ------------------------- |
| 샘플 수 (`sample_count`)  | 178                       |
| 특성 수 (`feature_count`) | 14 (13 features + target) |
| 클래스 수 (`class_count`)  | 3 (0, 1, 2)               |

### ✅ 통계 및 분포

* `value_counts()` → 클래스별 개수 확인
* `groupby("quality")["alcohol"].mean()` → 알코올 평균 비교
* `df["malic_acid"].mean()`, `std()` → 평균/표준편차 계산
* 특정 조건 비율 예시:

  ```python
  (df["color_intensity"] >= 10).mean() * 100
  ```

---

## 📍 2. 시각화를 통한 데이터 탐색

### ✅ 상관관계 Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(14, 12))
sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0)
plt.title("하삼각 상관계수 히트맵")
plt.show()
```

### ✅ 특성별 분포 탐색

```python
fig, ax = plt.subplots(figsize=(18, 5), ncols=3)

# 1️⃣ 세로형 히스토그램
sns.histplot(data=df, y="flavanoids", kde=True, color="steelblue", ax=ax[0])

# 2️⃣ 클래스별 히스토그램
sns.histplot(data=df, x="flavanoids", hue="quality", kde=True, element="step", palette="rocket", ax=ax[1])

# 3️⃣ 2D 히스토그램
sns.histplot(data=df, x="flavanoids", y="total_phenols", hue="quality", bins=20, palette="mako", ax=ax[2])
```

### ✅ 산점도

```python
sns.scatterplot(
    data=df, x="flavanoids", y="total_phenols",
    hue="quality", palette="rocket", s=60, edgecolor="none"
)
```

---

## 📍 3. sklearn을 통한 데이터 전처리

### ✅ Train/Test 분할 + 표준화

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop("quality", axis=1).values
y = df["quality"].values
y[y == 0] = 0
y[y != 0] = 1  # 이진 분류로 변환

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
```

---

## 📍 4. sklearn을 통한 모델 학습 및 검증

### ✅ 로지스틱 회귀 모델 학습

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_norm, y_train)
y_pred = clf.predict(X_test_norm)
```

### ✅ 평가 결과

```python
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**ConvergenceWarning**

> 반복 수(`max_iter`)가 부족하거나 정규화가 제대로 되지 않아 수렴하지 않을 때 발생.
> → 해결: `max_iter=1000` 이상, `StandardScaler` 적용.

### ✅ ROC-AUC 시각화

```python
y_score = clf.predict_proba(X_test_norm)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
auc = roc_auc_score(y_test, y_score)

plt.plot(fpr, tpr, label=f'ROC-AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## 📍 5. sklearn을 통한 교차검증 (Cross-Validation)

### ✅ 5-Fold F1-score 검증

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

f1_scores = cross_val_score(pipe, X, y, cv=5, scoring="f1")
print("Average F1-score (CV):", f1_scores.mean())
```

> 교차검증은 데이터 분할 편향을 줄이고 **모델의 일반화 성능**을 객관적으로 평가하기 위함입니다.

---

## 📍 6. sklearn을 통한 PCA 분석 (차원 축소 & 군집)

### ✅ PCA + K-Means 시각화

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

fig, ax = plt.subplots(figsize=(12, 4), ncols=2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="tab10", ax=ax[0])
ax[0].set_title("K-Means Clustering on PCA Space")
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="rocket", ax=ax[1])
ax[1].set_title("True Labels on PCA Space")
plt.suptitle("KMeans Clustering with PCA")
plt.show()
```

> 🔍 **해석:**
>
> * PCA는 다차원 데이터를 2D로 투영해 구조를 시각화.
> * K-Means 군집 결과가 실제 라벨과 얼마나 일치하는지 확인.
> * 완벽히 일치하지 않아도 **데이터의 내재된 분리 가능성**을 시각적으로 파악할 수 있음.

---

# ✅ 전체 요약

| 단계     | 주요 학습 포인트                            |
| ------ | ------------------------------------ |
| 데이터 탐색 | Wine 데이터셋 로딩, 통계·시각화, 상관관계           |
| 전처리    | `train_test_split`, `StandardScaler` |
| 지도학습   | 로지스틱 회귀를 통한 이진 분류, ROC-AUC 평가        |
| 교차검증   | `cross_val_score`로 F1-score 평균 산출    |
| 비지도학습  | PCA 차원축소 + K-Means 군집화 시각화           |

