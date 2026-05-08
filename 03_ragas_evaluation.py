import json
import os
import sys
import warnings
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from qa_pairs import QA_PAIRS


# ── 1. Python 3.12 Type Definitions ─────────────────────────────────────────
class RagRecord(TypedDict):
    question: str
    reference: str
    answer: str
    contexts: list[str]


# ── 2. Environment Setup ────────────────────────────────────────────────────
def setup_environment() -> None:
    """Load environment variables and suppress noisy warnings."""
    warnings.filterwarnings("ignore")
    load_dotenv()
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "day22-langsmith-lab")


# ── 3. Prompt Templates ─────────────────────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_V1), ("human", "{question}")]
)

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_V2), ("human", "{question}")]
)

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 4. RAG Core Functions ───────────────────────────────────────────────────
def build_vectorstore(embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """Build and return a FAISS vector store from the knowledge base."""
    kb_path = Path("data/knowledge_base.txt")
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

    text = kb_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    return FAISS.from_texts(chunks, embeddings)


def collect_rag_outputs(
    vectorstore: FAISS, prompt_version: str, llm: ChatGoogleGenerativeAI
) -> list[RagRecord]:
    """Execute the RAG pipeline over all QA pairs using LCEL."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PROMPTS[prompt_version]

    # Best Practice: Construct the LCEL chain once outside the loop
    chain = prompt | llm | StrOutputParser()

    results: list[RagRecord] = []
    total_q = len(QA_PAIRS)

    print(f"\nRunning {total_q} questions with prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, start=1):
        question = qa["question"]

        docs: list[Document] = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        ctx_str = "\n\n".join(contexts)

        # Invoke the pre-built chain
        answer = chain.invoke({"context": ctx_str, "question": question})

        results.append(
            {
                "question": question,
                "reference": qa["reference"],
                "answer": answer,
                "contexts": contexts,
            }
        )
        print(f"  [{i:02d}/{total_q}] {question[:60]}...")

    return results


# ── 5. Ragas Evaluation ─────────────────────────────────────────────────────
def build_ragas_dataset(rag_results: list[RagRecord]) -> EvaluationDataset:
    """Map the RAG results into the latest Ragas v0.2.x SingleTurnSample format."""
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval(
    rag_results: list[RagRecord],
    version: str,
    llm_eval: ChatGoogleGenerativeAI,
    emb_eval: GoogleGenerativeAIEmbeddings,
) -> dict[str, float]:
    """Run Ragas metrics and return aggregated scores."""
    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    # Ragas evaluate natively handles LangChain models
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_eval,
        embeddings=emb_eval,
    )

    # Best Practice: The new Ragas v0.2.x Result object allows direct key access
    # to aggregate float scores. Manual numpy mean aggregation is no longer necessary.
    scores = {metric.name: float(result[metric.name]) for metric in metrics}
    return scores


# ── 6. Main Execution ───────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation (Google GenAI)")
    print("=" * 60)

    setup_environment()

    # Pipeline Models
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

    # Collect Data
    v1_results = collect_rag_outputs(vectorstore, "v1", llm)
    v2_results = collect_rag_outputs(vectorstore, "v2", llm)

    # Evaluation Models (Isolated from the generation pipeline)
    llm_eval = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL_NAME", "gemini-3.1-flash-lite-preview"),
    )

    emb_eval = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL_NAME", "gemini-embedding-2"),
    )


    # Run Evaluations
    v1_scores = run_ragas_eval(v1_results, "v1", llm_eval, emb_eval)
    v2_scores = run_ragas_eval(v2_results, "v2", llm_eval, emb_eval)

    # Print Comparison Table
    print("\n📊 Comparison Table: V1 vs V2")
    print("-" * 45)
    for metric_name in v1_scores.keys():
        s1, s2 = v1_scores[metric_name], v2_scores[metric_name]
        print(f"  {metric_name:20s}: V1={s1:.4f}  V2={s2:.4f}")

    # Save Report
    report = {"prompt_v1_scores": v1_scores, "prompt_v2_scores": v2_scores}
    report_path = Path("data/ragas_report.json")

    # Ensure the parent directory exists before writing
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n💾 Saved {report_path}")


if __name__ == "__main__":
    main()
