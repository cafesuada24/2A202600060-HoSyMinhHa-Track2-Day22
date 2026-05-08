import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS

# ── 1. Python 3.12 Type Aliases ─────────────────────────────────────────────
# Using the new `type` statement introduced in Python 3.12
type RagChain = Runnable[str, str]


# ── 2. Environment Setup ────────────────────────────────────────────────────
def setup_environment() -> None:
    """Load environment variables and set defaults for LangSmith."""
    load_dotenv()
    
    # Use setdefault to avoid overwriting existing system env vars
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "day22-langsmith-lab")
    # Note: GOOGLE_API_KEY and LANGCHAIN_API_KEY are picked up automatically 
    # by the underlying SDKs if they exist in the .env file.


# ── 3. Factory Functions for LLM and Embeddings ─────────────────────────────
def get_llm() -> ChatGoogleGenerativeAI:
    """Initialize the Google Gemini model."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL_NAME", "gemini-3.1-flash-lite-preview"),
        # The SDK automatically infers google_api_key from os.environ["GOOGLE_API_KEY"]
    )

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize Google Generative AI embeddings."""
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL_NAME", "gemini-embedding-2"),
    )


# ── 4. Build Vector Store ───────────────────────────────────────────────────
def build_vectorstore(kb_path: Path, embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """Build and return a FAISS vector store from a text file."""
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")
        
    # Explicit encoding is a best practice when reading text
    text = kb_path.read_text(encoding="utf-8")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    print(f"Split knowledge base into {len(chunks)} chunks.")
    return FAISS.from_texts(chunks, embeddings)


# ── 5. Build the RAG Chain ──────────────────────────────────────────────────
def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(
    vectorstore: FAISS, 
    llm: ChatGoogleGenerativeAI
) -> tuple[RagChain, VectorStoreRetriever]:
    """Construct the LCEL RAG chain and return it along with the retriever."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the context below to answer accurately.\n\nContext:\n{context}"),
        ("human",  "{question}"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever


# ── 6. Traced Query Function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain: RagChain, question: str) -> str:
    """Execute a traced query against the RAG chain."""
    return chain.invoke(question)


# ── 7. Main Execution ───────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline (Google GenAI)")
    print("=" * 60)

    setup_environment()

    # Define paths using pathlib
    kb_path = Path("data/knowledge_base.txt")
    
    try:
        embeddings = get_embeddings()
        vectorstore = build_vectorstore(kb_path, embeddings)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    llm = get_llm()
    chain, _ = build_rag_chain(vectorstore, llm)

    for i, question in enumerate(SAMPLE_QUESTIONS, start=1):
        answer = ask(chain, question)
        
        # Format strings cleanly and truncate safely
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {answer[:100]}...\n")

    project_name = os.getenv('LANGCHAIN_PROJECT', 'Unknown Project')
    print(f"✅ {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{project_name}'")

if __name__ == "__main__":
    main()
