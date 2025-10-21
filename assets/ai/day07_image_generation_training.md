
# day07 [실습] 이미지 생성 및 평가와 모델 학습

## 🧩 개요
이번 실습에서는 **Stable Diffusion**을 이용해 이미지를 생성하고,  
**CLIP 모델**을 통해 생성 이미지의 의미적 유사도를 평가한 뒤,  
생성된 데이터를 사용해 **ResNet18 모델**을 학습시키는 과정을 진행하였습니다.

---

## 1️⃣ Stable Diffusion 기반 이미지 생성
- **라이브러리:** `diffusers`, `torch`
- **모델:** `runwayml/stable-diffusion-v1-5`
- **주요 파라미터**
  - `guidance_scale`: 프롬프트 반영 강도 (기본값 7.5)
  - `num_inference_steps`: 노이즈 제거 단계 수 (기본값 50)
  - `num_images_per_prompt`: 한 번에 생성할 이미지 수

```python
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
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

---

## 2️⃣ CLIP을 이용한 이미지-텍스트 유사도 평가
- **모델:** `openai/clip-vit-base-patch32`
- **입력:** 생성된 이미지 + 텍스트 라벨 리스트
- **출력:** 이미지와 각 텍스트 간의 유사도 확률

```python
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
```

💡 CLIP은 이미지와 텍스트를 동일한 임베딩 공간으로 투사하여  
서로의 의미적 유사도를 계산하는 멀티모달 모델입니다.

---

## 3️⃣ ResNet50을 통한 사전학습 모델 평가
- **모델:** `ResNet50_Weights.IMAGENET1K_V2`
- **전처리:** Resize(224), Normalize(mean, std)
- **출력:** ImageNet Top-5 예측 클래스

```python
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet50.eval()
with torch.no_grad():
    output = resnet50(img_tensor)
probs = torch.nn.functional.softmax(output, dim=1)
```

---

## 4️⃣ Stable Diffusion을 활용한 합성 데이터셋 생성
- **클래스 예시**
  - fox → 붉은 여우 수채화
  - dog → 골든리트리버 수채화
- 생성 데이터 저장 경로: `data/train/{class}` 및 `data/test/{class}`

```python
result = pipe(
    prompt,
    negative_prompt=neg_prompt,
    guidance_scale=7.5,
    num_inference_steps=50,
    num_images_per_prompt=2,
)
```

---

## 5️⃣ Transfer Learning 기반 ResNet18 학습
- **전이학습(Transfer Learning)**: 사전학습된 가중치를 고정하고,  
  새로운 출력층만 학습하는 *Linear Probing* 기법 적용

### (1) 가중치 고정
```python
for p in model.parameters():
    p.requires_grad = False
```

### (2) 출력층 교체
```python
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))  # 2개 클래스
```

### (3) 손실함수 및 옵티마이저
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
```

### (4) 학습 루프
```python
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

---

## ✅ 결과 요약
| 단계 | 주요 모델 | 목적 | 출력 |
|------|------------|------|------|
| 이미지 생성 | Stable Diffusion | 프롬프트 기반 합성 이미지 생성 | 수채화 여우/강아지 이미지 |
| 유사도 평가 | CLIP | 이미지-텍스트 의미 유사도 계산 | 유사도 확률값 |
| 사전학습 모델 평가 | ResNet50 | 생성 이미지 분류 정확도 확인 | Top-5 클래스 |
| 모델 학습 | ResNet18 (전이학습) | 합성 데이터 기반 분류 학습 | fox / dog 예측 |

---

## 📘 학습 포인트 정리
- Stable Diffusion으로 **데이터 생성 자동화** 가능성을 체험
- CLIP을 통해 **멀티모달 유사도 평가**의 개념 이해
- ResNet18 기반 **Transfer Learning 학습 구조** 확립
- 이미지 생성 → 평가 → 학습의 **엔드투엔드(End-to-End)** 흐름 완성

---

**작성자:** 숙제 미성 Transfer Learning 실습 세션  
**파일명:** day07_image_generation_training.md
