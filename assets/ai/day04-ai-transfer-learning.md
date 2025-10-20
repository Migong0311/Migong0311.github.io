
# 🔥 PyTorch를 활용한 MLP 구현하기

## 📘 MLP (Multi-Layer Perceptron) 기본 개념

* **MLP**는 가장 기본적인 인공신경망 구조로,
  입력층(Input) → 은닉층(Hidden) → 출력층(Output)으로 구성됩니다.
* 각 층에서는 `Linear(가중치 연산)` → `ReLU(비선형성)` → `Dropout(일부 뉴런 비활성화)` 순으로 데이터를 처리합니다.

### ✅ 구현 구조

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

---

## 🧠 객체에 ()를 걸면 어떻게 될까?

* `model = MLP(...)`
* `output = model(x)`
  → 사실은 내부적으로 **`model.__call__()`** → **`model.forward()`**가 호출되는 구조입니다.
* 즉, `()` 연산자는 모델의 **순전파(forward pass)** 를 수행합니다.

---

## 🧩 실습: 숫자 판독기 만들기 (MLP 학습 루프)

### 1️⃣ 학습용 함수

```python
def train_one_epoch(model, loader, optimizer, device):
    model.train()                     # 학습 모드
    running_loss, correct, total = 0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        y_targets = yb.squeeze(1).long()
        logits = model(xb)
        optimizer.zero_grad()
        loss = F.cross_entropy(logits, y_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_targets).sum().item()
        total += xb.size(0)

    return running_loss / total, correct / total
```

### 2️⃣ 평가용 함수

```python
def evaluate(model, loader, device):
    model.eval()                      # 추론 모드
    running_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():             # 미분 비활성화
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            y_targets = yb.squeeze(1).long()
            logits = model(xb)
            loss = F.cross_entropy(logits, y_targets)

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_targets).sum().item()
            total += xb.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(y_targets.cpu())

    return running_loss / total, correct / total
```

---

## ⚙️ model.train() vs model.eval()

| 모드                  | Dropout/BatchNorm 동작       | 목적                 |
| ------------------- | -------------------------- | ------------------ |
| **`model.train()`** | Dropout 활성화 (무작위로 뉴런 일부 끔) | 학습 중 일반화 성능 향상     |
| **`model.eval()`**  | Dropout/BN 고정, 평균·분산 사용    | 예측(추론) 시 일관된 결과 보장 |

> 💡 즉, `train()`은 “학습용”, `eval()`은 “테스트용 모드” 전환 스위치입니다.

---

## ⏳ 학습이 길어지면…?

* **과적합(Overfitting)** 발생 위험이 있습니다.
  → Validation loss가 증가하면 조기 종료(Early Stopping) 고려
* **Dropout**, **Learning Rate Scheduler**, **정규화** 등을 통해 완화합니다.
* GPU 사용 시 **학습 시간 단축 + batch 크기 최적화**가 중요합니다.

---

## 🔍 sklearn.classification에 있는 모듈로도 계산이 될까요?

네, 가능합니다.
PyTorch로 학습한 결과(`preds`, `targets`)를 넘파이 배열로 변환해
`sklearn.metrics`의 함수를 이용하면 됩니다.

```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(targets_cat, preds_cat))
print(classification_report(targets_cat, preds_cat))
```

---

## 💬 전체 흐름 요약

| 단계                      | 내용                                            |
| ----------------------- | --------------------------------------------- |
| ① 데이터 전처리               | `StandardScaler`로 정규화, Tensor 변환              |
| ② Dataset/DataLoader 구성 | `TensorDataset` → `DataLoader(batch_size)`    |
| ③ 모델 정의                 | `MLP(nn.Module)` 구성 (Linear + ReLU + Dropout) |
| ④ 학습 루프                 | `train_one_epoch()`으로 파라미터 업데이트               |
| ⑤ 검증 루프                 | `evaluate()`로 성능 확인                           |
| ⑥ 성능 평가                 | `sklearn.metrics`로 F1-score, 정확도 확인           |

---

## ✅ 핵심 포인트 정리

* MLP는 **입력-은닉-출력층으로 구성된 가장 단순한 신경망 구조**
* PyTorch의 핵심 학습 구조는
  **DataLoader → Model → Optimizer → Loss → Backpropagation**
* `model.train()` / `model.eval()` 로 모드를 명시해야 Dropout/BN이 정상 동작
* 학습 후 결과를 `sklearn` 모듈로 손쉽게 평가 가능


