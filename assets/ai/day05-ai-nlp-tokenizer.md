
# 🧠 토크나이저 / 워드 임베딩

## 1.1 Tokenizer 학습

**핵심 개념**

* 문장을 단어(토큰) 단위로 분리하여, 모델이 이해할 수 있는 숫자 시퀀스로 변환하는 과정입니다.

**질문 예시**

* “토크나이저는 어떤 기준으로 문장을 나누나요?”
* “BPE(Byte Pair Encoding) 방식과 WordPiece의 차이는 무엇인가요?”

---

## 1.2 토크나이저를 이용한 토큰 ID 시퀀스 변환

**핵심 개념**

* 학습된 토크나이저로 문장을 숫자 ID 시퀀스로 변환 (`encode`)
* 다시 숫자를 문장으로 복원 (`decode`)

**질문 예시**

* “`encode()`와 `decode()`는 각각 어떤 역할을 하나요?”
* “특수 토큰([CLS], [SEP], [PAD])은 어떤 상황에서 사용되나요?”

---

## 1.3 임베딩 벡터

**핵심 개념**

* 토큰 ID를 고정 길이의 **연속 벡터**로 바꿔주는 층 (e.g., `nn.Embedding`)
* 단어 간 의미적 유사도를 수치적으로 표현할 수 있게 함.

**질문 예시**

* “임베딩 벡터는 왜 필요한가요?”
* “원-핫 인코딩과 임베딩의 차이는 무엇인가요?”
* “`nn.Embedding`의 `vocab_size`와 `embedding_dim`은 각각 무엇을 의미하나요?”

---

# 🔁 RNN / LSTM

**핵심 개념**

* 순차적 데이터를 처리하기 위한 대표적인 순환 신경망 구조
* LSTM은 RNN의 장기 의존성 문제(vanishing gradient)를 해결함.

**질문 예시**

* “RNN이 시계열 데이터를 잘 다루는 이유는 뭔가요?”
* “LSTM의 ‘cell state’는 어떤 역할을 하나요?”
* “`nn.RNN`과 `nn.LSTM`의 입력·출력 형태 차이를 설명해주세요.”

---

# 🎯 Attention Mechanism

**핵심 개념**

* 입력 시퀀스 전체를 요약하지 않고, 각 시점별 **가중치(attention weight)** 를 계산해 중요한 정보에 집중하는 메커니즘.
* `LuongAttention`, `BahdanauAttention` 등이 대표적.

**질문 예시**

* “Attention의 핵심 아이디어는 무엇인가요?”
* “Luong 어텐션과 Bahdanau 어텐션의 차이점은?”
* “어텐션 가중치는 어떻게 계산되나요?”

---

# 🤗 HuggingFace 라이브러리 활용

**핵심 개념**

* 사전 학습된 모델(`BERT`, `GPT`, `T5`, `RoBERTa` 등)을 간편하게 불러와 사용할 수 있는 라이브러리.
* Tokenizer, Model, Trainer 등 고수준 API 제공.

**질문 예시**

* “`from_pretrained()` 메서드는 어떤 역할을 하나요?”
* “HuggingFace의 `pipeline`을 이용하면 어떤 일을 자동화할 수 있나요?”
* “`tokenizer`, `model`, `trainer`는 각각 어떤 역할을 하나요?”

---

# 🧩 아키텍처별 모델 다뤄보기 (Encoder / Decoder)

**핵심 개념**

* **Encoder 모델**: 입력을 인코딩(요약)하는 역할 (e.g., BERT)
* **Decoder 모델**: 입력을 바탕으로 출력 시퀀스를 생성 (e.g., GPT, T5)
* **Seq2Seq 구조**: Encoder와 Decoder를 연결해 문장 변환/번역 등에 활용

**질문 예시**

* “Encoder와 Decoder의 구조적 차이는 무엇인가요?”
* “BERT와 GPT는 각각 어떤 방식으로 텍스트를 처리하나요?”
* “Seq2Seq 구조에서 Attention이 어디에 들어가나요?”

