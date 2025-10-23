# 🧠 (실습 문제) 4-1 RAG 기반 Customer Service AI 에이전트 개발

**Retrieval-Augmented Generation (RAG) 기반 고객 서비스 AI Agent**

---

## 🩵 1. 실습 개요

본 실습은 **LangChain + Upstage Solar 모델**을 활용하여
AI 온라인 서점(Customer Service Bot)을 RAG 방식으로 구현하는 실습입니다.

### 🎯 목표

* **RAG 파이프라인 전체 구축 흐름**을 이해하고 직접 구현.
* **Vector Store + Retriever + LLM 결합 구조**를 익혀 도메인 기반 질의응답 가능 모델 설계.
* AI의 **Hallucination(환각)** 문제를 줄이고, **지식 기반 응답**을 만들 수 있도록 하는 것.

### 📍 핵심 키워드

* **LangChain**, **Retriever**, **Vector Store**, **Embeddings**, **Chunking**,
  **LLM Chain**, **Tool Integration**, **Agent System**

---

## ⚙️ 2. 실습 목적 및 배경

### (1) 왜 RAG인가?

기존 LLM은 학습된 시점 이후의 정보나 도메인 내부 문서를 알지 못함.
→ 외부 지식 베이스(Vector Store)에 문서를 저장하고,
질문 시 관련 내용을 검색 후 답변에 반영함으로써 **지식 확장형 챗봇**을 구축.

### (2) Customer Service 시나리오

* “AI 온라인 서점”을 예로 하여, 고객이 **배송 정책, 주문 상태, 반품 규정** 등을 물어보면
  → 에이전트가 내부 정책 문서를 검색하고, 근거 기반으로 답변하도록 설계.

---

## 💡 3. 학습 목표

1. **RAG 파이프라인 구성요소 이해**

   * Document Load → Chunking → Embedding → Vector Store 저장 → Retriever 생성 → LLM과 결합.
2. **텍스트 처리 능력 향상**

   * Tokenization, Chunking 기법, Recursive Splitter의 동작 원리 이해.
3. **LangChain의 Agent System 이해**

   * LLM이 Tool을 통해 검색/계산 등 외부 기능을 호출하는 구조 학습.
4. **LLM as Tool User**

   * Tool을 정의하고, Agent가 어떤 상황에서 특정 Tool을 자동 선택하게 하는 구조 실습.

---

## 🧩 4. RAG 구성 단계별 정리

### 🔹 (1) 환경 설정

#### 주요 설치 패키지

```bash
pip install langchain langchain_community langchain_upstage chromadb tiktoken
```

#### 환경 변수 관리

* `.env` 파일에 API Key 저장 (`UPSTAGE_API_KEY`)
* `load_dotenv()` 로 불러와서 코드 내에서 안전하게 사용.

#### 보안 포인트

* API Key는 코드에 직접 입력 금지.
* `.gitignore`를 통해 GitHub 업로드 시 `.env` 제외 필수.

---

### 🔹 (2) 텍스트 데이터 준비 및 처리

#### ✅ 텍스트 로딩 (TextLoader)

```python
from langchain_community.document_loaders import TextLoader
loader = TextLoader("./shipping_policy.txt")
documents = loader.load()
```

* 문서 단위로 불러오며, `page_content`와 `metadata`를 포함한 리스트 반환.

#### ✅ Chunking (RecursiveCharacterTextSplitter)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)
chunks = splitter.split_documents(documents)
```

* 긴 문서를 일정 크기로 분할하여 검색 성능 향상.
* **chunk_overlap**은 문맥 연속성을 보장하기 위해 일부 텍스트를 겹쳐서 자름.
* 일반적으로 **500~1000자**, overlap은 **10~20%** 추천.

#### ✅ Tokenization (tiktoken)

```python
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(text)
```

* LLM 입력 단위를 측정하여 적절한 chunk 크기를 조정하는 데 사용.

---

### 🔹 (3) 벡터 스토어(Vector Store) 구축

#### ✅ Embedding 생성

```python
from langchain_upstage import UpstageEmbeddings
embedding_model = UpstageEmbeddings()
```

* 문장을 고정 길이 벡터로 변환 (semantic embedding).
* 의미 유사도를 기준으로 문서 검색 가능.

#### ✅ ChromaDB를 이용한 Vector Store 구성

```python
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(chunks, embedding_model)
```

#### ✅ Retriever 생성

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
```

* 입력 질문과 유사한 문서 Top-k를 반환.

---

### 🔹 (4) Retriever Tool 생성

```python
from langchain.tools.retriever import create_retriever_tool

shipping_policy_retriever_tool = create_retriever_tool(
    retriever,
    "shipping_policy_search_tool",
    "Searches and returns information about the AI online bookstore's shipping policy."
)

agent_tools = [shipping_policy_retriever_tool]
```

> ✅ Tool은 LangChain Agent가 사용할 수 있는 “외부 기능(도구)”입니다.
> 예를 들어 “배송 정책에 대해 알려줘” → Agent가 Retriever Tool을 자동 호출해 답변을 보강.

---

## 🧠 5. LLM Chain 구성

### (1) ChatUpstage 모델 불러오기

```python
from langchain_upstage import ChatUpstage
llm = ChatUpstage()
```

### (2) 프롬프트 템플릿 정의

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "모든 답변은 존댓말로 해주세요."),
    ("human", "프랑스의 수도는 어딘지 알아요?"),
    ("ai", "알고 있습니다. 파리입니다."),
    ("human", "일본의 수도는 어딘지 알아요?"),
    ("ai", "알고 있습니다. 도쿄입니다."),
    ("human", "그렇다면, 한국은?")
])
```

* System 메시지로 **역할(role)** 설정.
* Few-shot Prompting으로 **예시 대화** 제공.
* 사용자 입력이 `"한국은?"`일 때 유사 패턴 학습.

### (3) Chain 구성 및 실행

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({})
print(result)
```

* `|` 연산자는 LangChain의 **Composable Chain 구조**를 의미.
* 최종적으로 **입력 → 프롬프트 → 모델 → 출력 파서**로 연결.

---

## 🛠️ 6. 사용자 정의 Tool 추가

### (1) 인사 응답 도구 (`@tool`)

```python
from langchain.tools import tool

@tool
def greet_customer(query: str) -> str:
    """고객의 인삿말을 감지하여 환영 메시지 반환"""
    greeting_keywords = ["안녕", "안녕하세요", "하이", "반가워요"]
    if any(k in query for k in greeting_keywords):
        return "안녕하세요! AI 온라인 서점입니다. 무엇을 도와드릴까요?"
    return ""
```

### (2) Agent Tool 리스트에 추가

```python
agent_tools.append(greet_customer)
```

→ 최종 Agent는 **검색용 Retriever Tool**과 **인사 응답 Tool**을 함께 사용 가능.

---

## 🧩 7. Agent 동작 시나리오

| 사용자 입력            | 에이전트의 행동                                            | 사용된 도구         |
| ----------------- | --------------------------------------------------- | -------------- |
| “안녕하세요!”          | `greet_customer()` 호출 → 인사 메시지 출력                   | greet_customer |
| “배송 기간이 어떻게 되나요?” | `shipping_policy_retriever_tool` 호출 → 관련 문서 검색 후 답변 | retriever tool |
| “책이 언제 도착하나요?”    | Retriever 검색 → 배송 정책 내용 기반 생성                       | retriever tool |

---

## 🧱 8. 시험 대비 핵심 정리

| 구분                     | 핵심 내용                     | 예시 키워드                           |     |         |
| ---------------------- | ------------------------- | -------------------------------- | --- | ------- |
| **Chunking**           | 문서 분할, overlap 설정 이유      | `RecursiveCharacterTextSplitter` |     |         |
| **Embedding**          | 문장을 벡터로 변환 → 의미 검색        | `UpstageEmbeddings`, `Chroma`    |     |         |
| **Retriever**          | 관련 문서 검색기                 | `.as_retriever()`                |     |         |
| **RAG**                | Retrieval + Generation 결합 | “검색 + 생성 기반 답변”                  |     |         |
| **LangChain Chain**    | 프롬프트 → 모델 → 파서 연결         | `prompt                          | llm | parser` |
| **Agent Tool**         | Agent가 호출 가능한 함수          | `create_retriever_tool`, `@tool` |     |         |
| **Few-shot Prompting** | 예시 대화 제공으로 출력 패턴 유도       | “파리입니다, 도쿄입니다”                   |     |         |
| **.env 관리**            | API Key 보안 유지             | `load_dotenv()`                  |     |         |
| **tiktoken**           | 토큰 단위로 청킹 효율화             | `"cl100k_base"`                  |     |         |
| **LLM as Agent**       | 상황에 따라 적절한 Tool 자동 선택     | `agent_tools` 리스트                |     |         |

---

## 🧩 9. 실습 결론

이 실습을 통해 다음 역량을 획득합니다:

1. **RAG 파이프라인 전 주기 구현 능력**
   (문서 로드 → 임베딩 → 벡터 스토어 → 검색기 → 생성기 결합)
2. **LangChain의 Agent System 구조 이해**
   (Tool 등록, Chain 구성, 자동 Tool 호출)
3. **도메인 기반 챗봇 설계 역량**
   (고객센터·FAQ 시스템에 직접 적용 가능)
4. **LLM 프롬프트 엔지니어링 및 구조적 설계 감각**

---

✅ **시험 포인트 요약**

* RAG의 정의: “검색(Retrieval) + 생성(Generation)” 구조
* Retriever의 역할: 질문과 가장 유사한 문서를 찾아줌
* Vector Store: 임베딩을 저장·검색하는 데이터베이스
* Chunking 이유: 문맥 단절 최소화 및 검색 효율성 확보
* Agent Tool 예시: `greet_customer`, `shipping_policy_retriever_tool`
* Prompt 구성요소: System / Human / AI / Few-shot
* LangChain 파이프라인: prompt → llm → parser
* `.env` 관리 및 보안
* tiktoken 역할: LLM 입력 토큰 단위 제어

---

**결론:**

> 본 실습은 단순한 대화형 챗봇이 아니라,
> “**내부 문서를 근거로 정확한 답변을 생성하는 RAG 기반 지능형 고객 응대 AI**”를
> 직접 설계하고 구현하는 과정을 다룹니다.
