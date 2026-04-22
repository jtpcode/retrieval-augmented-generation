import re
import numpy as np
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "superstore_sales"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"
TOP_K = 8          # Number of chunks to retrieve per query
MMR_LAMBDA = 0.5    # Balance between relevance (1.0) and diversity (0.0)
MAX_HISTORY_TURNS = 5  # Number of conversational turns to keep in memory

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


MONTHS_PATTERN = (
    r"\b(January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\b"
)

SUB_CATEGORIES = (
    "Bookcases|Chairs|Furnishings|Tables|Appliances|Art|Binders|Envelopes|"
    "Fasteners|Labels|Paper|Storage|Supplies|Accessories|Copiers|Machines|Phones"
)


def extract_metadata_filters(query):
    # Extracts structured metadata filters from the query.

    filters = {}

    year_matches = re.findall(r'\b(201[4-7])\b', query)
    if len(year_matches) == 1:
        # $contains matches chunks where year="2016" or year="2015, 2016, 2017"
        filters["year"] = {"$contains": year_matches[0]}

    region_matches = re.findall(r'\b(Central|East|South|West)\b', query, re.IGNORECASE)
    if len(region_matches) == 1:
        filters["region"] = {"$contains": region_matches[0].capitalize()}

    category_matches = re.findall(r'\b(Furniture|Office Supplies|Technology)\b', query, re.IGNORECASE)
    if len(category_matches) == 1:
        filters["category"] = {"$contains": category_matches[0]}

    month_matches = re.findall(MONTHS_PATTERN, query, re.IGNORECASE)
    if len(month_matches) == 1:
        filters["month"] = {"$contains": month_matches[0].capitalize()}

    sub_cat_matches = re.findall(rf'\b({SUB_CATEGORIES})\b', query, re.IGNORECASE)
    if len(sub_cat_matches) == 1:
        filters["sub_category"] = {"$contains": sub_cat_matches[0].capitalize()}

    return filters


def _cosine_similarity(a, b):
    #Compute cosine similarity between two vectors.
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom > 0 else 0.0


def _apply_mmr(
    query_embedding,
    candidate_docs,
    candidate_metas,
    candidate_embeddings,
    top_k,
    lambda_param = MMR_LAMBDA,
):
    # Selects top_k diverse chunks using Maximal Marginal Relevance.

    selected_indices = []
    selected_embeddings = []
    remaining = list(range(len(candidate_docs)))

    while len(selected_indices) < top_k and remaining:
        best_idx = None
        best_score = float("-inf")

        for i in remaining:
            sim_to_query = _cosine_similarity(query_embedding, candidate_embeddings[i])
            if selected_embeddings:
                max_sim_to_selected = max(
                    _cosine_similarity(candidate_embeddings[i], s)
                    for s in selected_embeddings
                )
            else:
                max_sim_to_selected = 0.0
            score = lambda_param * sim_to_query - (1 - lambda_param) * max_sim_to_selected

            if score > best_score:
                best_score = score
                best_idx = i

        selected_indices.append(best_idx)
        selected_embeddings.append(candidate_embeddings[best_idx])
        remaining.remove(best_idx)

    return [
        {"document": candidate_docs[i], "metadata": candidate_metas[i], "distance": 0.0}
        for i in selected_indices
    ]


def retrieve_context(query, top_k = TOP_K):
    """
    Retrieves diverse, relevant chunks via metadata filtering + MMR:
      1. Extract metadata filters from the query (year, region, category, month, sub-category).
      2. Fetch a larger candidate pool via filtered similarity search.
      3. Fall back to unfiltered search if the filter yields no results.
      4. Apply MMR to select top_k diverse chunks from the candidates.
    """
    query_embedding = embedding_model.encode([query]).tolist()[0]

    filters = extract_metadata_filters(query)
    where = None
    if len(filters) == 1:
        where = filters
    elif len(filters) > 1:
        where = {"$and": [{k: v} for k, v in filters.items()]}

    n_candidates = min(top_k * 4, collection.count())

    if where:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            where=where,
            include=["documents", "metadatas", "embeddings"],
        )
        if not results["documents"][0]:
            where = None

    if not where:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "embeddings"],
        )

    return _apply_mmr(
        query_embedding,
        results["documents"][0],
        results["metadatas"][0],
        results["embeddings"][0],
        top_k,
    )


def build_messages(
    query,
    chunks,
):
    # Constructs the full message list for the Ollama chat API.

    system_prompt = (
        "You are an expert data analyst specializing in retail sales analytics.\n"
        "You answer questions about a Superstore sales dataset covering 2014-2017,\n"
        "with transactions across categories (Furniture, Office Supplies, Technology),\n"
        "regions (Central, East, South, West), customers, and products.\n\n"
        "Guidelines:\n"
        "- Be concise and data-driven. Cite specific numbers from the context.\n"
        "- Reference retrieved data chunks when making claims (e.g., 'According to the data...').\n"
        "- For trend or comparison questions, highlight the most significant differences.\n"
        "- If the context is insufficient to fully answer, state that clearly.\n"
        "- Never sum, estimate, or fabricate totals not present in the context.\n"
    )

    # Format retrieved chunks with index and metadata tags
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        meta_parts = []
        for key in ("year", "month", "category", "region", "sub_category"):
            if key in chunk["metadata"]:
                meta_parts.append(f"{key}={chunk['metadata'][key]}")
        meta_tag = f" [{', '.join(meta_parts)}]" if meta_parts else ""
        context_lines.append(f"[{i}]{meta_tag} {chunk['document']}")


    context_block = "\n".join(context_lines)

    user_message = (
        f"Use ONLY the following data:\n{context_block}\n\n"
        f"Question: {query}"
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    return messages


def query_rag(
    query,
    top_k = TOP_K,
):
    # Executes a full RAG query: retrieve -> prompt -> generate.

    chunks = retrieve_context(query, top_k=top_k)
    messages = build_messages(query, chunks)
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    answer = response.message.content

    return answer, chunks


def main():
    # Chat loop:

    print("Superstore RAG System (2014-2017)")
    print("Commands: 'quit' to exit")

    while True:
        try:
            query = input("\nQuery: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit"):
            print("Goodbye!")
            break

        print("[Thinking...]")

        answer, chunks = query_rag(query)

        print(f"\nAnswer: {answer}")

        # Print a compact source summary
        sources = sorted({c["metadata"].get("source", "unknown") for c in chunks})
        print(f"\n[{len(chunks)} chunks retrieved from: {', '.join(sources)}]")


if __name__ == "__main__":
    main()
