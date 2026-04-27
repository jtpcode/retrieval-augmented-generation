import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import re
import time

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "superstore_sales"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"
TOP_K = 5

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

def get_collection():
	client = chromadb.PersistentClient(path=CHROMA_PATH)
	return client.get_collection(name=COLLECTION_NAME)

def retrieve(query, collection, model, top_k=TOP_K):
    query_emb = model.encode([query]).tolist()[0]
    filters = extract_metadata_filters(query)
    where = filters if filters else None

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas"],
        where=where
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return list(zip(docs, metas))

def build_prompt(query, docs_and_metas):
	context_lines = []
     
    # Example of a line: [1] Total sales in 2016 were $1.2M (year=2016, region=East)
	for i, (doc, meta) in enumerate(docs_and_metas, 1):
		meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else ""
		context_lines.append(f"[{i}] {doc} {meta_str}")

	context = "\n".join(context_lines)

	prompt = (
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
		f"{context}\n\n"
		f"Question: {query}\n"
		"Answer:"
	)

	return prompt

def ask_llm(prompt):
	response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
	return response.message.content

def main():
    print("Superstore RAG")
    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = get_collection()

    while True:
        query = input("\nAsk a question (or 'quit'): ").strip()
        if query.lower() == "quit":
            break
        docs_and_metas = retrieve(query, collection, model)
        prompt = build_prompt(query, docs_and_metas)
        print("[Thinking...]")
        start = time.time()
        answer = ask_llm(prompt)
        elapsed = time.time() - start
        print(f"\nAnswer: {answer}\n")
        print(f"(Time taken: {elapsed:.2f} seconds)")

        # Print sources of retrieved chunks for transparency
        sources = sorted({meta.get("source", "unknown") for _, meta in docs_and_metas})
        print(f"\n[{len(docs_and_metas)} chunks retrieved from: {', '.join(sources)}]")

if __name__ == "__main__":
	main()
