
# 🧠 [실습] Transfer Learning 기반의 CNN 모델 학습 및 ViT 활용 이미지 분류

---

## 📌 1️⃣ 데이터 전처리 (Data Transformation)

* 이미지 데이터를 신경망 입력 크기(224×224)로 맞추고 정규화(Normalization) 수행
* CIFAR-10 평균(mean), 표준편차(std)를 사용
* 학습(train)과 테스트(test)용 변환기를 각각 정의

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

> 🧩 **목적:**
> 모델 학습이 안정적으로 이루어지도록 입력 이미지를 정규화 및 통일된 크기로 변환합니다.

---

## 📌 2️⃣ 데이터셋 및 데이터로더 구성

* CIFAR-10 데이터셋(50,000장 train, 10,000장 test)을 로드
* `DataLoader`를 이용해 미니배치(batch) 단위로 모델에 공급

```python
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
```

> 🧩 **목적:**
> 데이터를 효율적으로 배치 단위로 공급하고, GPU 학습에 적합한 형태로 구성합니다.

---

## 📌 3️⃣ 모델 수정 (ResNet-18 기반 Transfer Learning)

* 사전학습된 **ResNet-18 (ImageNet Pretrained)** 모델 사용
* 마지막 분류층(`fc`)을 CIFAR-10에 맞게 **10개의 클래스**로 교체

```python
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
```

> 🧩 **목적:**
> 기존 모델의 특성 추출기(Feature Extractor)는 그대로 사용하고,
> 마지막 분류기만 교체하여 새로운 데이터셋에 맞게 학습합니다.

---

## 📌 4️⃣ 파라미터 동결 (Freeze)

* **선형 프로빙(Linear Probing)**:
  사전 학습된 파라미터는 그대로 두고 **마지막 레이어(fc)** 만 학습하도록 설정

```python
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False
```

> 🧩 **목적:**
> 이미 학습된 특징(feature)을 그대로 사용하고,
> 새 데이터셋에 대한 분류기만 학습하여 빠르고 안정적인 성능 향상.

---

## 📌 5️⃣ 손실 함수와 옵티마이저 정의

* **손실 함수 (Loss):** CrossEntropyLoss
* **옵티마이저:** SGD, 학습률 0.001
* **fc 레이어만 업데이트되도록 설정**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
```

---

## 📌 6️⃣ 모델 학습 루프

* 옵티마이저 초기화 → 순전파 → 손실 계산 → 역전파 → 가중치 업데이트
* 각 epoch마다 평균 손실 출력

```python
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f"[Epoch {epoch+1}/{num_epochs}] 평균 훈련 손실: {avg_loss:.4f}")
```

> 🧩 **핵심 로직:**
> `forward → loss → backward → step()`
> 을 통해 모델이 CIFAR-10의 각 클래스(비행기, 자동차, 고양이 등)를 학습합니다.

---

## 📌 7️⃣ 모델 평가 (Accuracy 계산)

* 테스트 데이터로 모델 성능 평가
* `torch.max()`를 이용해 예측 클래스 도출 및 정확도 계산

```python
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in tqdm(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"테스트 데이터 정확도: {accuracy:.2f}%")
```

> 🧩 **목적:**
> 학습되지 않은 데이터셋에서의 **일반화 성능** 평가.

---

## 📌 8️⃣ 데이터 증강 (Data Augmentation)

* 이미지에 **RandomCrop + RandomHorizontalFlip** 추가
* 일반화 성능 향상에 도움

```python
train_transform_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
```

---

## 📌 9️⃣ 미세 조정 (Fine-Tuning)

* 모든 파라미터의 동결 해제 (`requires_grad=True`)
* 모델 전체를 학습시켜 CIFAR-10에 완전히 적응하도록 조정

```python
for param in model.parameters():
    param.requires_grad = True
```

* 학습률을 낮게 설정한 옵티마이저로 전체 레이어 업데이트

```python
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
```

---

## 📌 🔟 학습률 스케줄러 적용

* 학습이 진행됨에 따라 학습률을 점차 감소시켜 안정적인 수렴 유도

```python
scheduler.step()
```

> 🧩 예시:
>
> ```python
> scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
> ```

---

## 📌 11️⃣ Vision Transformer (ViT) 모델 로드

* HuggingFace Hub에서 **사전학습된 ViT 모델**과 **전처리기** 불러오기

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification

model_name = "nateraw/vit-base-patch16-224-cifar10"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
vit_model = ViTForImageClassification.from_pretrained(model_name)
vit_model.to(device)
```

> 🧩 **목적:**
> Transformer 기반 구조(ViT)를 사용하여 CNN보다 더 효율적인 시각적 패턴 학습을 수행합니다.

---

## 📌 12️⃣ Hugging Face 파이프라인을 이용한 예측

* `pipeline("image-classification")`을 통해 손쉽게 이미지 분류 수행

```python
from transformers import pipeline

clf = pipeline("image-classification", model=model_name, device=device)
preds = clf("cat_image.jpg")
print(preds)
```

📈 **출력 예시**

```python
[{'label': 'cat', 'score': 0.9982},
 {'label': 'dog', 'score': 0.0011}]
```

> 🧩 **목적:**
> 복잡한 전처리 과정을 자동화하고,
> 단 한 줄의 코드로 이미지 분류 결과를 얻을 수 있음.

---

# 🎯 **최종 요약**

| 단계    | 핵심 개념                   | 주요 목적                       |
| ----- | ----------------------- | --------------------------- |
| 1~2   | 데이터 전처리 및 로딩            | 이미지 정규화 및 DataLoader 구성     |
| 3~5   | 모델 수정 및 동결              | ResNet18을 CIFAR-10용으로 변환    |
| 6~7   | 학습 및 평가                 | CrossEntropyLoss, SGD 기반 훈련 |
| 8     | 데이터 증강                  | 일반화 향상                      |
| 9~10  | Fine-Tuning + Scheduler | 모델 전체 재학습 + 안정적 수렴          |
| 11~12 | Vision Transformer 적용   | 사전학습된 ViT 모델로 이미지 분류        |

---

✅ **결론:**
이번 실습은

> “사전학습된 모델(ResNet18, ViT)을 활용하여 CIFAR-10 분류를 수행하고,
> 선형 프로빙 → 미세조정(Fine-Tuning) → Transformer 확장까지”
> 전 과정을 경험하는 **Transfer Learning의 핵심 실습**입니다.
