
# 🧭 **Day 02 — 데이터 정규화 및 선형대수 기반 해법 (시험 대비 정리)**

---

## 🧮 1️⃣ 데이터 정규화 및 준비

### 🔹 (1) 연속형 / 범주형 변수 구분

| 구분                    | 설명                                | 예시                      |
| ----------------------- | ----------------------------------- | ------------------------- |
| **연속형(Continuous)**  | 수치형 값으로 크기·차이를 계산 가능 | `age`, `price`, `income`  |
| **범주형(Categorical)** | 라벨 값, 구분만 가능                | `gender`, `region`, `day` |

> ✅ `df.select_dtypes('object')`로 범주형 변수 선택 가능

```python
categorical_cols = df.select_dtypes('object').columns
```

---

### 🔹 (2) 표준화(Standardization)

* **정의:**
  각 특성을 평균 0, 표준편차 1로 변환 (Z-score scaling)
* **이유:**
  변수의 단위 차이로 인해 모델이 한 특성에 과도하게 영향받는 것을 방지

📘 **공식**

![alt text](/assets/img/Standardization.png)

> 🔹 회귀나 거리 기반 모델(`KNN`, `SVM`, `PCA`)에서 매우 중요

---

### 👨‍💻 (3) 표준화 실습 코드

```python
mean = X.mean(axis=0)
sigma = X.std(axis=0)

sigma_safe = np.where(sigma == 0, epsilon, sigma)
X_norm = (X - mean) / sigma_safe
```

> ⚠️ **주의:** 표준편차가 0인 경우(즉, 값이 일정한 feature)는 `epsilon`으로 대체해야 함

---

### 🧠 (4) `to_numpy()` 메서드

| 목적            | 설명                      |
| --------------- | ------------------------- |
| 행렬 연산 수행  | pandas → numpy 변환       |
| 머신러닝 학습용 | DataFrame을 matrix로 전환 |

```python
X_array = df[['col1', 'col2']].to_numpy()
```

> 💡 `numpy` 배열은 `@` 연산자(행렬 곱) 지원으로 수식 구현 용이

---

## 📘 2️⃣ 선형대수 기반 해법 (Linear Regression)

---

### 🔹 (1) 정규방정식 (Normal Equation)

#### ✅ 개념

* **미분 없이 한 번에 해를 구하는 방법**
* MSE(평균제곱오차)를 최소화하는 해:

![alt text](/assets/img/Normal_Equation.png)

> 🔹 행렬 역연산을 직접 수행하므로 데이터 크기가 클수록 연산량 급증

---

#### ⚠️ 역행렬이 존재하지 않을 때

* ( X^T X )가 **singular(비가역)** 일 경우 역행렬 불가
* 해결책 → **SVD(특이값 분해)** 또는 **ridge regularization**

---

#### 💻 실습 코드

```python
XT_X = X_b.T @ X_b
XT_y = X_b.T @ y
theta = np.linalg.inv(XT_X) @ XT_y
```

> 💡 `@` 는 행렬 곱, `np.linalg.inv()`는 역행렬 계산

---

### 💡 절편항(bias) 추가와 hstack의 의미

| 개념            | 설명                                             |
| --------------- | ------------------------------------------------ |
| **절편항**      | y = a·x + b에서 b(절편)을 반영하기 위해 1열 추가 |
| **np.hstack()** | 수평 결합으로 [1, x₁, x₂, x₃] 형태 생성          |

```python
X_b = np.hstack([np.ones((m, 1)), X])
```

|   원본 X   | 추가 열(1) |    결합 결과    |
| :--------: | :--------: | :-------------: |
| x₁, x₂, x₃ |     1      | [1, x₁, x₂, x₃] |

---

### 🔹 (2) 최소제곱법 (Least Squares)

#### ✅ 정의

“모든 데이터에 대한 오차 제곱의 합이 최소가 되도록 θ 찾기”

#### 💻 수치적 안정해법

```python
theta_lstsq, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
```

#### 🔍 MSE 계산

```python
y_pred = X_b @ theta_lstsq
mse = np.mean((y_pred - y) ** 2)
```

> 🔹 `np.linalg.lstsq()`는 역행렬 불가 시에도 안정적으로 최소제곱해 계산

---

### 🔹 (3) SVD를 이용한 해 구하기

#### ✅ 개념

* SVD(특이값 분해)는 역행렬이 존재하지 않아도 **유사역행렬(pseudo-inverse)** 계산 가능
* 수학적으로 안정적이며 수치 오차에 강함

![alt text](/assets/img/SVD.png)



---

#### 💻 실습 코드

```python
U, S, Vt = np.linalg.svd(X_b, full_matrices=False)
S_plus = np.diag(1 / S)
theta_svd = Vt.T @ S_plus @ U.T @ y
```

> 🔹 실제 구현 시 `np.linalg.pinv(X)` 와 동일한 결과
> 🔹 역행렬 불가한 경우에도 안전하게 회귀계수 계산 가능

---

## ⚙️ 3️⃣ 경사하강법 (Gradient Descent)

#### ✅ 개념

* 해를 직접 구하지 않고, **손실 감소 방향으로 θ를 반복 갱신**
* 반복(iteration)으로 최적 해 근사

![alt text](/assets/img/Gradient_Descent.png)



---

#### 💻 기본 구현 예시

```python
theta = np.zeros(n+1)
alpha = 0.01
iterations = 1000
loss_history = []

for i in range(iterations):
    y_pred = X_b @ theta
    error = y_pred - y.flatten()
    mse = np.mean(error ** 2)
    loss_history.append(mse)
    gradient = (2/m) * (X_b.T @ error)
    theta = theta - alpha * gradient
```

| 용어              | 의미                                      |
| ----------------- | ----------------------------------------- |
| α (learning rate) | 학습률, 너무 크면 발산 / 작으면 수렴 느림 |
| gradient          | 손실 함수의 기울기                        |
| loss_history      | 학습 과정에서 손실 추적                   |

> 🔹 시험포인트: **α 값 조절**이 수렴 안정성의 핵심

---

## 🧮 4️⃣ 미니배치 + Gradient Accumulation

| 개념              | 설명                                                  |
| ----------------- | ----------------------------------------------------- |
| **미니배치 학습** | 데이터를 batch 단위로 학습 → 메모리 절약, 노이즈 완화 |
| **Accumulation**  | 여러 batch의 gradient 평균 후 업데이트                |

#### 💻 구현 흐름

```python
for epoch in range(epochs):
    grad_accum = np.zeros_like(theta)
    accum_count = 0

    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        X_batch = X_b[start:end]
        y_batch = y[start:end].ravel()

        y_pred = X_batch @ theta
        error = y_pred - y_batch
        grad = (2.0 / len(X_batch)) * (X_batch.T @ error)

        grad_accum += grad
        accum_count += 1

        if accum_count == accumulate_steps or end == m:
            theta -= alpha * (grad_accum / accum_count)
            grad_accum = np.zeros_like(theta)
            accum_count = 0
```

> 🔹 **accumulate_steps**: 여러 미니배치의 gradient를 합산해 평균 적용
> 🔹 대용량 데이터 학습 시 매우 효율적

---

## 📊 5️⃣ MSE 평가 및 시각화

![alt text](/assets/img/MSE.png)

#### 💻 시각화 함수

```python
def plot_prediction(y_true, y_pred):
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    sns.lineplot(
        x=[y.min(), y.max()],
        y=[y.min(), y.max()],
        linestyle="--", color="red"
    )
```

> 🔹 점들이 대각선에 가까울수록 예측 성능 우수
> 🔹 MSE는 “평균 제곱 오차(Mean Squared Error)”를 의미

---

## 📈 6️⃣ 종합 요약표 (시험 대비)

| 단계            | 핵심 내용                  | 주요 함수                        |
| --------------- | -------------------------- | -------------------------------- |
| **데이터 준비** | 연속형/범주형 구분, 표준화 | `select_dtypes`, `to_numpy`      |
| **정규방정식**  | 역행렬 직접 계산           | `np.linalg.inv`                  |
| **최소제곱법**  | 안정적 수치 해법           | `np.linalg.lstsq`                |
| **SVD 해법**    | 역행렬 불가 시 대안        | `np.linalg.svd`                  |
| **경사하강법**  | 반복적 손실 최소화         | gradient, α(learning rate)       |
| **미니배치 GD** | 효율적 학습, 메모리 절약   | `batch_size`, `accumulate_steps` |

---

## 🧾 시험 자주 출제 포인트 정리

| 키워드         | 핵심 포인트                            |
| -------------- | -------------------------------------- |
| **표준화**     | 평균 0, 분산 1로 스케일 조정 (Z-score) |
| **정규방정식** | θ = (XᵀX)⁻¹Xᵀy, 역행렬 존재 조건 주의  |
| **SVD**        | 역행렬이 불가능할 때 안정적 대안       |
| **경사하강법** | α값이 너무 크면 발산, 너무 작으면 느림 |
| **Mini-Batch** | 대용량 데이터에서 자주 사용되는 방식   |
| **MSE**        | 회귀 문제의 대표 손실함수              |
| **np.hstack**  | 절편항(1열) 추가 시 사용               |
| **to_numpy()** | DataFrame → ndarray 변환 (연산 필수)   |

---

> 💡 **한줄 요약:**
> “Day 02은 데이터를 표준화한 후, 정규방정식·SVD·경사하강법 등
> 다양한 **선형대수 기반 회귀 해법**을 비교·이해하는 실습입니다.”


