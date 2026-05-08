import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

from qa_pairs import SAMPLE_QUESTIONS

# ── 1. Python 3.12 Type Aliases ─────────────────────────────────────────────
type PromptDict = dict[str, ChatPromptTemplate]

# ── 2. Prompt Definitions ───────────────────────────────────────────────────
PROMPT_V1_NAME = "day22-rag-prompt-v1"
PROMPT_V2_NAME = "day22-rag-prompt-v2"

SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Environment & Setup ──────────────────────────────────────────────────
def setup_environment() -> None:
    """Load environment variables safely without overwriting system defaults."""
    load_dotenv()
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "day22-langsmith-lab")


def build_vectorstore(embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """Read knowledge base and construct a FAISS vector store."""
    kb_path = Path("data/knowledge_base.txt")
    
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")
        
    text = kb_path.read_text(encoding="utf-8")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    return FAISS.from_texts(chunks, embeddings)


# ── 4. Prompt Hub Operations ────────────────────────────────────────────────
def push_prompts_to_hub(client: Client) -> None:
    """Push both prompt versions to LangSmith Hub."""
    try:
        client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise answers")
        print("✅ Pushed V1")
    except Exception as e:
        print(f"⚠️  V1 push failed: {e}")

    try:
        client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured answers")
        print("✅ Pushed V2")
    except Exception as e:
        print(f"⚠️  V2 push failed: {e}")


def pull_prompts_from_hub(client: Client) -> PromptDict:
    """Pull prompt versions from LangSmith Hub, falling back to locals on failure."""
    prompts: PromptDict = {}
    
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}'")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}'")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2

    return prompts


# ── 5. Traced Query & Routing ───────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """Determine prompt version via MD5 hash for consistent A/B routing."""
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(
    retriever: VectorStoreRetriever, 
    llm: ChatGoogleGenerativeAI, 
    prompt: ChatPromptTemplate, 
    question: str, 
    version: str
) -> dict[str, str]:
    """Execute and trace a RAG query using A/B tested prompts."""
    # Retrieve
    docs: list[Document] = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Generate
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return {"question": question, "answer": answer, "version": version}


# ── 6. Main Execution ───────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing (Google GenAI)")
    print("=" * 60)

    setup_environment()
    
    # Implicit API key loading relies on the environment variables directly
    client = Client()
    
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL_NAME", "gemini-3.1-flash-lite-preview"),
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL_NAME", "gemini-embedding-2"),
    )
    
    try:
        vectorstore = build_vectorstore(embeddings)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"

        # Execute the routed request
        result = ask_ab(retriever, llm, prompts[version_key], question, version_tag)
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

    print(f"\n📊 Routing Summary: V1={v1_count}, V2={v2_count}")

if __name__ == "__main__":
    main()
