
# 🧭 데이터 정규화 및 선형대수 기반 해법 정리

---

## 🧮 1. 데이터 정규화 및 준비

### 🔹 1.1 연속형 / 범주형 변수 분류하기

* **연속형 변수(Continuous)**: 수치형 값 (예: `age`, `price`, `income`)
* **범주형 변수(Categorical)**: 범주(label) 값 (예: `gender`, `region`, `day`)

> `categorical_cols` 리스트를 따로 정의하여 구분할 수 있습니다.

```python
categorical_cols = df.select_dtypes('object').columns
```

---

### 🧩 1.2 표준화(Standardization)의 정의

* 각 특성(feature)을 **평균 0, 표준편차 1**로 스케일링하는 과정입니다.
* 수식:

![alt text](/assets/img/Standardization.png)

* 이유: 모든 특성이 같은 단위로 비교되도록 하여, 회귀나 거리 기반 모델이 특정 변수에 치우치지 않게 만듭니다.

---

### 👨‍💻 1.3 실습: 표준화 진행해보기

```python
mean = X.mean(axis=0)
sigma = X.std(axis=0)

sigma_safe = np.where(sigma == 0, epsilon, sigma)  # 0으로 나누기 방지
X_norm = (X - mean) / sigma_safe
```

> ⚙️ `sigma_safe`는 분모가 0이 되지 않도록 **ε(아주 작은 수)** 로 대체한 안전한 표준편차 벡터입니다.

---

### 🧠 1.4 `to_numpy()` 메서드란?

* pandas DataFrame을 numpy 배열로 변환하는 메서드입니다.
  → 회귀 모델에서 **행렬 연산**을 수행하기 위해 필수!

```python
X_array = df[['col1', 'col2']].to_numpy()
```

---

## 📘 2. 선형대수를 이용한 해법 (Linear Regression)

---

### 🔹 2.1 정규방정식 (Normal Equation)

#### ✅ 개념

* 선형회귀 해를 **미분 없이 한 번에 구하는 방법**입니다.
* 최소제곱오차(MSE)를 최소화하는 해:

![alt text](/assets/img/Normal_Equation.png)


#### ⚠️ 역행렬이 항상 존재하지는 않음

* ( X )가 정사각 행렬이 아닐 수도 있고,
* ( X^T X )가 **역행렬 불가능(singular)** 한 경우도 있음 → 해결책으로 **SVD** 사용.

#### 🧩 실습

```python
XT_X = X_b.T @ X_b
XT_y = X_b.T @ y
theta = np.linalg.inv(XT_X) @ XT_y
```

---

### 💡 절편항(bias)과 hstack의 의미

#### ✅ 절편항

* 회귀식 ( y = a x + b ) 에서 **b(절편)** 을 추가하기 위해
  ( X ) 행렬에 **1의 열(column)** 을 추가합니다.

#### ✅ np.hstack

* 수평으로 배열을 이어 붙이는 함수
  → 절편항을 추가할 때 사용.

```python
X_b = np.hstack([np.ones((m, 1)), X])
```

|   원래 X   | np.ones((m,1)) |  → hstack 결과  |
| :--------: | :------------: | :-------------: |
| x₁, x₂, x₃ |       1        | [1, x₁, x₂, x₃] |

---

### 🔹 2.2 최소제곱법 (Least Squares)

#### ✅ 수학적 정의

* “모든 데이터에 대해 오차 제곱의 합이 최소가 되는 θ 찾기”
* 수치적으로 안정적인 `np.linalg.lstsq()` 활용:

```python
theta_lstsq, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
```

#### 🔍 MSE 계산

```python
y_pred = X_b @ theta_lstsq
mse = np.mean((y_pred - y) ** 2)
```

---

### 🔹 2.3 SVD를 이용한 최소제곱 해 구하기

#### ✅ 개념

* **SVD(특이값 분해)** 는 역행렬이 존재하지 않아도
  ( X^T X ) 의 안정적인 역을 구할 수 있게 합니다.
* 수식:

![alt text](/assets/img/SVD.png)


#### 🧩 실습 코드

```python
U, S, Vt = np.linalg.svd(X_b, full_matrices=False)
S_plus = np.diag(1 / S)
theta_svd = Vt.T @ S_plus @ U.T @ y
```

---

## ⚙️ 3. 경사하강법 (Gradient Descent)과 손실 계산

#### ✅ 개념

* 해를 직접 계산하지 않고, **반복적으로 손실을 줄이는 방향으로 θ를 업데이트**하는 방법입니다.
* 기본 업데이트 식:

![alt text](/assets/img/Gradient_Descent.png)


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

---

## 🧮 추가 심화: 미니배치 + Gradient Accumulation

#### ✅ 핵심 아이디어

* 데이터를 한 번에 쓰지 않고, 작은 덩어리(batch)로 나눠 학습.
* 여러 배치를 모아서 한 번에 업데이트 → 메모리 절약 + 안정성 향상.

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
        error  = y_pred - y_batch
        grad   = (2.0 / len(X_batch)) * (X_batch.T @ error)

        grad_accum += grad
        accum_count += 1

        if accum_count == accumulate_steps or end == m:
            theta -= alpha * (grad_accum / accum_count)
            grad_accum = np.zeros_like(theta)
            accum_count = 0
```

---

## 📊 MSE 평가 및 시각화

#### ✅ MSE (Mean Squared Error)

![alt text](/assets/img/MSE.png)


#### 💻 시각화 함수

```python
def plot_prediction(y_true, y_pred):
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    sns.lineplot(x=[y.min(), y.max()],
                 y=[y.min(), y.max()],
                 linestyle="--", color="red")
```

---

## 📈 정리 요약

| 단계        | 핵심 내용                  | 주요 메서드                       |
| ----------- | -------------------------- | --------------------------------- |
| 데이터 준비 | 연속형/범주형 구분, 표준화 | `.select_dtypes()`, `.to_numpy()` |
| 정규방정식  | 역행렬 기반 해 구하기      | `np.linalg.inv()`                 |
| 최소제곱법  | 안정적 수치 해법           | `np.linalg.lstsq()`               |
| SVD 해법    | 역행렬 불가 시 대안        | `np.linalg.svd()`                 |
| 경사하강법  | 반복적 손실 최소화         | `for loop`, `gradient update`     |
| 미니배치 GD | 효율적 학습                | `batch_size`, `accumulate_steps`  |

