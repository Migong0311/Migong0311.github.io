
# 🧠 **Day 04 — PyTorch 기반 MLP 실습 (시험 대비용 정리)**

---

## 📘 1️⃣ MLP (Multi-Layer Perceptron) 기본 개념

| 항목      | 설명                                                                           |
| --------- | ------------------------------------------------------------------------------ |
| 정의      | 입력층 → 은닉층(1개 이상) → 출력층으로 구성된 **기본 신경망 구조**             |
| 주요 연산 | `Linear → ReLU → Dropout`                                                      |
| 역할      | 비선형 함수(활성화 함수)로 복잡한 관계를 학습                                  |
| 특징      | 완전연결층(fully connected), 데이터의 구조 정보(공간적/시계열)는 고려하지 않음 |

### ✅ 기본 구조 예시

```python
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes)
        )
    def forward(self, x):
        return self.net(x)
```

> 🔹 `nn.Linear` : 입력 → 출력으로 가중치 곱 + 편향(bias)
> 🔹 `nn.ReLU` : 비선형성 부여
> 🔹 `nn.Dropout(p)` : 학습 시 p 비율만큼 뉴런을 비활성화 (과적합 방지)

---

## ⚙️ 2️⃣ `model(x)`의 내부 동작

| 호출                                           | 내부적으로 수행되는 메서드               |
| ---------------------------------------------- | ---------------------------------------- |
| `model(x)`                                     | `model.__call__(x)` → `model.forward(x)` |
| 즉, 괄호 `()`는 **forward propagation**을 의미 |                                          |

> ✅ forward: 입력 → 예측(logits)
> ✅ backward: 손실 기반으로 역전파(gradient 계산)

---

## 🧩 3️⃣ 학습 및 평가 함수 구조

### 🔹 (1) 학습 루프 — `train_one_epoch()`

```python
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb.squeeze(1).long())
        loss.backward()
        optimizer.step()
```

| 단계                    | 의미                        |
| ----------------------- | --------------------------- |
| `optimizer.zero_grad()` | 이전 step의 gradient 초기화 |
| `loss.backward()`       | 역전파(gradient 계산)       |
| `optimizer.step()`      | 파라미터 업데이트           |

> 🔹 시험포인트: `zero_grad()` 생략 시 → 이전 batch의 gradient가 누적되어 잘못된 업데이트 발생

---

### 🔹 (2) 평가 루프 — `evaluate()`

```python
def evaluate(model, loader, device):
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = logits.argmax(dim=1)
```

| 항목              | 설명                                     |
| ----------------- | ---------------------------------------- |
| `model.eval()`    | Dropout 비활성화, BN의 이동평균 사용     |
| `torch.no_grad()` | gradient 계산 비활성화(속도 ↑, 메모리 ↓) |
| `argmax(dim=1)`   | 예측된 클래스 번호 추출                  |

> 🔹 시험포인트: `train()` vs `eval()`의 Dropout 차이 반드시 숙지

---

## 🧠 4️⃣ `train()` vs `eval()` 비교 요약

| 모드        | Dropout                  | BatchNorm             | 사용 시점 |
| ----------- | ------------------------ | --------------------- | --------- |
| **train()** | 활성화 (무작위 비활성화) | 배치별 평균/분산 사용 | 학습 단계 |
| **eval()**  | 비활성화                 | 저장된 이동 평균 사용 | 추론 단계 |

> ✅ 오답주의: `model.eval()`은 gradient를 끄지 않습니다.
> (→ 반드시 `torch.no_grad()`와 함께 써야 함)

---

## 🧩 5️⃣ 과적합(Overfitting) 방지 전략

| 전략                    | 설명                                               |
| ----------------------- | -------------------------------------------------- |
| Dropout                 | 일부 뉴런 무작위 비활성화                          |
| L2 정규화               | 가중치 크기에 패널티 부여 (`weight_decay`)         |
| Early Stopping          | 검증 손실이 증가하면 학습 중단                     |
| Learning Rate Scheduler | 학습률 점진적 감소 (`StepLR`, `ReduceLROnPlateau`) |
| Data Augmentation       | 입력 데이터 다양화                                 |

> 🔹 시험포인트: Dropout은 학습 시만 적용되고, 평가 시엔 비활성화

---

## 🧮 6️⃣ 학습 후 평가 (sklearn 활용)

```python
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(targets, preds))
print(classification_report(targets, preds))
```

| 함수                      | 설명                               |
| ------------------------- | ---------------------------------- |
| `confusion_matrix()`      | 클래스별 예측 정오표               |
| `classification_report()` | Precision / Recall / F1-score 출력 |

> 🔹 실제 딥러닝 모델 평가 시 sklearn과 함께 자주 사용

---

## ⚡ 7️⃣ 전체 학습 흐름 요약

| 단계              | 설명                              | 주요 함수                             |
| ----------------- | --------------------------------- | ------------------------------------- |
| ① 데이터 전처리   | 표준화 후 Tensor 변환             | `StandardScaler`, `TensorDataset`     |
| ② DataLoader 구성 | 미니배치 제공                     | `DataLoader(batch_size)`              |
| ③ 모델 정의       | 입력-은닉-출력층 구성             | `nn.Linear`, `nn.ReLU`, `nn.Dropout`  |
| ④ 학습 루프       | 순전파 → 손실 → 역전파 → 업데이트 | `loss.backward()`, `optimizer.step()` |
| ⑤ 평가 루프       | 추론 및 정확도 계산               | `model.eval()`, `torch.no_grad()`     |
| ⑥ 성능 평가       | 정확도, F1-score                  | `classification_report()`             |

---

## 🧾 8️⃣ 추가 개념 정리 (시험 자주 출제)

| 개념                    | 설명                                                       |
| ----------------------- | ---------------------------------------------------------- |
| **Batch size**          | 한 번의 업데이트에 사용되는 데이터 개수                    |
| **Epoch**               | 전체 데이터셋이 한 번 학습되는 단위                        |
| **Optimizer 종류**      | `SGD`, `Adam`, `RMSProp` 등 (Adam이 가장 자주 사용)        |
| **Loss Function**       | 분류는 `CrossEntropyLoss`, 회귀는 `MSELoss`                |
| **Gradient Descent**    | 손실 최소화를 위한 파라미터 갱신 알고리즘                  |
| **Activation Function** | 비선형성 부여 (`ReLU`, `Sigmoid`, `Tanh`)                  |
| **Sequential 모델**     | 순차적으로 layer를 쌓는 가장 단순한 구조 (`nn.Sequential`) |

---

## ✅ 핵심 요약 (시험 대비 포인트)

1. `model(x)`은 내부적으로 `__call__()` → `forward()` 실행
2. `train()` 모드에서는 Dropout 활성화 / `eval()` 모드에서는 비활성화
3. `zero_grad()`, `backward()`, `step()`의 순서 중요
4. `torch.no_grad()`는 평가 시 gradient 비활성화용
5. Overfitting 방지 방법 3가지 이상 서술 가능해야 함
6. 학습 후 sklearn의 `confusion_matrix`, `classification_report`로 평가
7. MLP는 CNN/RNN보다 단순하지만 구조 이해의 기본

---

> 💡 **한줄 요약:**
> “MLP는 PyTorch 신경망의 기본 구조로, Linear + ReLU + Dropout으로 구성되며
> 학습 루프는 zero_grad → forward → loss → backward → step 순으로 반복된다.”

