# 🧠 day06 Transfer Learning 기반 CNN & ViT 이미지 분류 종합 정리

---

## 1️⃣ 데이터 전처리 (Transform & Normalize)

**핵심**

* 입력 크기 통일(224×224), 텐서 변환, **정규화(평균/표준편차)**
* CIFAR-10의 mean/std 적용 (또는 ImageNet 통계 사용 시 모델 일관성 고려)

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

**시험 포인트**

* 정규화 통계(어느 데이터셋 기준인지) 일관성
* train/test 모두 동일 정규화
* 배치 차원: `NCHW` (PyTorch)

---

## 2️⃣ 데이터셋 & DataLoader

**핵심**

* CIFAR-10 (train 50k / test 10k)
* `shuffle=True`는 **train만**
* `num_workers`, `pin_memory` 설정으로 I/O 병목 완화

```python
trainset = torchvision.datasets.CIFAR10(..., transform=train_transform)
testset  = torchvision.datasets.CIFAR10(..., transform=test_transform)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
testloader  = DataLoader(testset,  batch_size=256, shuffle=False)
```

**시험 포인트**

* **shuffle**의 목적(분포 편향 감소)
* **batch_size** 증가 시 메모리/수렴 속도 트레이드오프

---

## 3️⃣ Transfer Learning: ResNet-18 수정

**핵심**

* ImageNet 사전학습 **백본은 유지**, `fc`만 10 class로 교체

```python
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
```

**시험 포인트**

* 사전학습 가중치의 장점(수렴 속도↑, 데이터 적을 때 효과↑)
* 마지막 레이어만 교체하는 이유(표현은 유지, 분류기만 적응)

---

## 4️⃣ 파라미터 동결 (Linear Probing)

**핵심**

* **특징 추출기 고정** + `fc`만 학습 → 빠르고 안정적

```python
for n, p in model.named_parameters():
    if "fc" not in n:
        p.requires_grad = False
```

**시험 포인트**

* Linear probing 목적 vs Fine-tuning 차이
* 동결 해제는 **미세조정 단계에서**

---

## 5️⃣ 손실 & 옵티마이저

**핵심**

* 분류는 **CrossEntropyLoss**
* Linear probing 단계: `model.fc.parameters()`만 업데이트

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
```

**시험 포인트**

* CrossEntropy는 **logits** 입력(softmax 미적용) 전제
* `momentum` 역할(진동 완화, 수렴 가속)

---

## 6️⃣ 학습 루프 (forward → loss → backward → step)

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
```

**시험 포인트(자주 틀리는 부분)**

* `optimizer.zero_grad()` 순서
* `model.train()` / `model.eval()` 전환
* `torch.no_grad()`는 **eval 시** 필수(메모리/속도)

---

## 7️⃣ 평가 (Accuracy)

```python
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
acc = 100 * correct / total
```

**시험 포인트**

* **일반화 성능**은 train이 아닌 **test/val**에서 측정
* 정확도 외에 **F1, Top-k** 등 지표도 상황에 따라 활용

---

## 8️⃣ 데이터 증강 (Augmentation)

**핵심**

* `RandomCrop(32, padding=4)`, `RandomHorizontalFlip` 등
* 과적합 완화, **일반화 성능 상승**

```python
train_transform_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

**시험 포인트**

* train에만 강한 증강 적용, test는 **deterministic** 전처리

---

## 9️⃣ Fine-Tuning (전체 미세 조정)

**핵심**

* 모든 파라미터 **동결 해제** 후 전체 업데이트
* 학습률은 probing보다 **작게**

```python
for p in model.parameters():
    p.requires_grad = True
optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
```

**시험 포인트**

* 작은 LR로 세밀 조정(특징 파괴 방지)
* 레이어별 다른 LR(Discriminative LR)도 가능

---

## 🔟 학습률 스케줄러

**핵심**

* 수렴 안정화를 위해 **주기적/조건적** LR 감소

```python
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
for epoch in range(num_epochs):
    ...
    scheduler.step()
```

**시험 포인트**

* `StepLR` vs `CosineAnnealingLR` vs `ReduceLROnPlateau` 차이
* 스케줄러 **호출 시점**(에폭 끝/지표 모니터링 후)

---

## 1️⃣1️⃣ Vision Transformer (ViT)

**핵심**

* **패치 분할** → **선형 임베딩** → **Transformer 인코더**
* CNN과 달리 **자연스러운 전역 시야** 확보, 대규모 사전학습에 강점

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
name = "nateraw/vit-base-patch16-224-cifar10"
feat = ViTFeatureExtractor.from_pretrained(name)
vit  = ViTForImageClassification.from_pretrained(name).to(device)
```

**시험 포인트**

* ViT 입력 크기(224), 패치 크기(16×16), **Position Embedding**
* 작은 데이터셋에선 **사전학습 + 미세조정**이 필수적

---

## 1️⃣2️⃣ HF `pipeline`으로 간편 추론

```python
from transformers import pipeline
clf = pipeline("image-classification", model=name, device=device)
preds = clf("cat_image.jpg")
```

**장점**

* 전처리/후처리 자동
* **프로토타이핑 속도↑**

**시험 포인트**

* 프로덕션/커스텀 훈련에는 **모델/토크나이저 직접 사용**이 더 유연

---

## 🧩 CNN vs ViT — 비교 요약

| 항목   | CNN(ResNet18)        | ViT                    |
| ---- | -------------------- | ---------------------- |
| 전처리  | ImageNet 통계 / 224 입력 | 224 입력 + 패치 분할(16)     |
| 장점   | 소규모 데이터 강함, 빠름       | 전역 컨텍스트, 대규모 사전학습에서 강력 |
| 단점   | 전역 정보 제한             | 데이터 적으면 과적합/불안정 가능     |
| 파인튜닝 | 상/하위 레이어 차등 LR 자주 사용 | 작은 LR + 긴 스케줄 권장       |

---

## ⚠️ 자주 하는 실수(시험 함정)

* `model.eval()`인데 `torch.no_grad()` 미사용 → 느리고 메모리 낭비
* 정규화 통계 불일치(ImageNet vs CIFAR) → 성능 급락
* 동결 상태에서 `model.parameters()`로 최적화 → **fc 외까지 갱신 위험**
* `scheduler.step()` 호출 타이밍 오류
* augmentation을 test에도 적용

---

## ✅ 체크리스트 (제출 전 1분 점검)

* [ ] 정규화 통계 일관성(Train/Test 동일)
* [ ] Linear probing → Fine-tuning 단계적 적용
* [ ] `train()/eval()` + `no_grad()` 정확히 사용
* [ ] Optimizer 파라미터 대상 정확히 지정
* [ ] 스케줄러/세이브/시드 고정(재현성)
* [ ] 평가는 **test/val** 기준으로 기록

---

### 한줄 결론

> **사전학습 백본**에 **Linear Probing → Fine-Tuning**을 단계적으로 적용하고,
> 필요 시 **ViT**로 확장하여 **전역 문맥**을 활용하면 CIFAR-10 분류에서 안정적으로 성능을 끌어올릴 수 있다.
