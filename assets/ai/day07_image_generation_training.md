
# 🧠 **Day07 이미지 생성 및 평가와 모델 학습 (시험 대비 정리)**

---

## 🧩 **개요**

본 실습은 **Stable Diffusion → CLIP → ResNet18**의 3단계 흐름을 통해
이미지를 “**생성–평가–학습**”하는 **End-to-End 생성형 AI 파이프라인**을 구성하는 과정입니다.

> 🔹 핵심 목표
>
> 1. **Stable Diffusion**으로 합성 이미지 생성
> 2. **CLIP**으로 이미지-텍스트 의미 유사도 평가
> 3. **ResNet18** Transfer Learning으로 학습

---

## 1️⃣ Stable Diffusion 기반 이미지 생성

### 🔹 주요 개념

* **Stable Diffusion**은 텍스트 프롬프트를 입력받아 이미지를 생성하는 **Latent Diffusion Model (LDM)**입니다.
* “텍스트 → 잠재공간(embedding) → 노이즈 제거 → 고해상도 이미지 복원” 순으로 작동합니다.

### 🔹 주요 파라미터

| 파라미터                    | 의미                            | 기본값 |
| ----------------------- | ----------------------------- | --- |
| `guidance_scale`        | 프롬프트 반영 강도 (높을수록 정확, 낮을수록 다양) | 7.5 |
| `num_inference_steps`   | 노이즈 제거 단계 수                   | 50  |
| `num_images_per_prompt` | 한 번에 생성할 이미지 수                | 1~4 |

```python
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
result = pipe(
    positive_prompt,
    negative_prompt=negative_prompt,
    guidance_scale=7.5,
    num_inference_steps=50,
    num_images_per_prompt=2,
)
```

> ⚙️ **Tip:** `negative_prompt`를 활용해 품질이 낮거나 원하지 않는 요소(“blurry”, “low quality”)를 배제할 수 있습니다.

---

## 2️⃣ CLIP 기반 이미지-텍스트 유사도 평가

### 🔹 개념

* **CLIP (Contrastive Language–Image Pretraining)**
  → 이미지와 텍스트를 **같은 임베딩 공간**에 매핑하고,
  두 표현 간의 **코사인 유사도**를 통해 의미적 일치도를 계산하는 모델입니다.

```python
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
```

| 출력                 | 설명                        |
| ------------------ | ------------------------- |
| `logits_per_image` | 이미지-텍스트 간 유사도 점수 (logits) |
| `probs`            | Softmax 정규화된 확률값 (0~1)    |

> 💡 **핵심 포인트:**
> CLIP은 “이 이미지가 어떤 라벨에 더 가깝나?”를 **자연어 수준**에서 평가할 수 있습니다.

---

## 3️⃣ ResNet50을 이용한 사전학습 모델 평가

### 🔹 목적

Stable Diffusion이 생성한 이미지가 **기존 분류 모델(ResNet50)**에서
합리적인 클래스로 인식되는지를 확인합니다.

```python
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet50.eval()
with torch.no_grad():
    output = resnet50(img_tensor)
probs = torch.nn.functional.softmax(output, dim=1)
```

| 단계                   | 처리 내용          |
| -------------------- | -------------- |
| Resize(224×224)      | 입력 크기 통일       |
| Normalize(mean, std) | ImageNet 통계 적용 |
| Softmax              | 클래스별 확률 계산     |

> ✅ **해석:**
> 상위 5개 클래스(top-5)의 확률이 높게 나올수록, 생성된 이미지가 자연스러운 패턴을 가짐을 의미합니다.

---

## 4️⃣ Stable Diffusion으로 합성 데이터셋 제작

### 🔹 목적

모델 학습용 **인공 데이터**를 자동 생성하기 위함.

| 클래스 예시 | 프롬프트                                          | 저장 경로            |
| ------ | --------------------------------------------- | ---------------- |
| fox    | “a watercolor painting of a red fox”          | `data/train/fox` |
| dog    | “a watercolor painting of a golden retriever” | `data/train/dog` |

```python
result = pipe(
    prompt,
    negative_prompt=neg_prompt,
    guidance_scale=7.5,
    num_inference_steps=50,
    num_images_per_prompt=2,
)
```

> ⚙️ 실제로는 2~4장의 이미지를 생성해 train/test 폴더 구조로 정리.

---

## 5️⃣ Transfer Learning 기반 ResNet18 학습

### (1) **Linear Probing (가중치 동결)**

```python
for p in model.parameters():
    p.requires_grad = False
```

### (2) **출력층 교체**

```python
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))
```

### (3) **손실 함수 및 옵티마이저**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
```

### (4) **학습 루프**

```python
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

> 💡 **핵심 원리:**
> 기존 사전학습된 Feature Extractor는 유지하고,
> 새 클래스(fox/dog)에 맞게 **마지막 Linear Layer만 학습**합니다.

---

## ✅ 결과 요약

| 단계      | 주요 모델                        | 핵심 목적              | 출력 결과          |
| ------- | ---------------------------- | ------------------ | -------------- |
| **1**   | Stable Diffusion             | 프롬프트 기반 합성 이미지 생성  | 수채화 여우/강아지 이미지 |
| **2**   | CLIP                         | 이미지-텍스트 의미 유사도 평가  | 유사도 확률         |
| **3**   | ResNet50                     | 사전학습 모델로 이미지 품질 검증 | Top-5 클래스      |
| **4~5** | ResNet18 (Transfer Learning) | 합성 데이터 기반 분류 학습    | fox/dog 예측     |

---

## 📘 학습 포인트 & 시험 대비 요약

| 개념                | 핵심 내용                 | 자주 나오는 질문                                |
| ----------------- | --------------------- | ---------------------------------------- |
| Stable Diffusion  | 텍스트→이미지 변환 LDM 구조     | “guidance_scale이 높으면?” → 정확도↑, 다양성↓      |
| CLIP              | 이미지·텍스트 공통 임베딩 공간     | “CLIP은 어떤 방식으로 유사도를 측정하나?” → 코사인 유사도     |
| Transfer Learning | Feature 고정 + 분류기만 재학습 | “Linear Probing과 Fine-Tuning 차이점은?”      |
| ResNet 구조         | 잔차 블록(Residual Block) | “skip connection의 효과는?” → gradient 소실 완화 |

---

## ⚙️ 한눈에 보는 전체 구조

```
[Stable Diffusion]
  ↓ (합성 이미지 생성)
[CLIP]
  ↓ (유사도 평가)
[ResNet18 Transfer Learning]
  ↓ (분류 학습 및 테스트)
```

---

## ✅ 핵심 요약

> “Stable Diffusion으로 합성 데이터를 만들고,
> CLIP으로 품질을 평가한 뒤,
> ResNet18 Transfer Learning으로 분류 모델을 학습하는
> **생성·평가·학습 통합 실습**이다.”




