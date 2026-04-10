import re
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="superstore_sales")

MONTHS = (
    "January|February|March|April|May|June|"
    "July|August|September|October|November|December"
)


def extract_metadata_from_line(line, source):
    meta = {"source": source}

    # Year
    if m := re.search(r'\b(201[4-7])\b', line):
        meta["year"] = int(m.group(1))

    # Month
    if m := re.search(MONTHS, line):
        meta["month"] = m.group(0)

    # Category
    if m := re.search(r'\b(Furniture|Office Supplies|Technology)\b', line, re.IGNORECASE):
        meta["category"] = m.group(1)

    # Sub-category: transactions end with ")", summaries end with ","
    if m := re.search(r'subcategory ([\w][\w ]+?)[),]', line):
        meta["sub_category"] = m.group(1)

    # Region
    if m := re.search(r'\b(Central|East|South|West)\b region', line, re.IGNORECASE):
        meta["region"] = m.group(1)

    # City and State from transaction lines: "from CITY, STATE (REGION region)"
    if m := re.search(r'from ([A-Za-z ]+), ([A-Za-z ]+?) \((?:Central|East|South|West) region\)', line):
        meta["city"] = m.group(1).strip()
        meta["state"] = m.group(2).strip()
    # State from state_summaries.txt: "In STATE state,"
    if m := re.search(r'In ([A-Z][a-zA-Z ]+) state,', line):
        meta["state"] = m.group(1)
    # City from city_summaries.txt: "In CITY city,"
    if m := re.search(r'In ([A-Z][a-zA-Z ]+) city,', line):
        meta["city"] = m.group(1)

    # Sales: "sales were $X", "sales of $X", "Sales: $X"
    if m := re.search(r'[Ss]ales(?:: |\s+(?:were|of)\s+)\$([\d,]+\.?\d*)', line):
        meta["sales"] = float(m.group(1).replace(",", ""))

    # Profit: "profit was $X", "profit of $X", "profit: $X"
    if m := re.search(r'[Pp]rofit(?:: |\s+(?:was|of)\s+)\$([\d,]+\.?\d*)', line):
        meta["profit"] = float(m.group(1).replace(",", ""))

    # Loss: "loss was $X", "loss of $X", "loss: $X"
    if m := re.search(r'[Ll]oss(?:: |\s+(?:was|of)\s+)\$([\d,]+\.?\d*)', line):
        meta["loss"] = float(m.group(1).replace(",", ""))

    return meta


# Process all .txt files line by line
print("Processing text files...")
text_files_dir = Path("text_files")
all_chunks = []
all_metadatas = []

for file in text_files_dir.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        all_chunks.append(line)
        all_metadatas.append(extract_metadata_from_line(line, file.name))

print(f"Total chunks created: {len(all_chunks)}")

# Generate IDs for the chunks
print("Generating IDs...")
ids = [f"chunk_{i}" for i in range(len(all_chunks))]

# Generate embeddings
print("Generating embeddings...")
embeddings = embedding_model.encode(all_chunks, show_progress_bar=True).tolist()

# Upsert into ChromaDB
# NOTE: The order of ids, documents, embeddings, and metadatas match because they are all based on the order of 'all_chunks',
# so they are correctly associated with each other when upserting into ChromaDB.

# NOTE: ChromaDB has a batch size limit of 5461
batch_size = 5000
print("Upserting into ChromaDB in batches...")

for i in range(0, len(all_chunks), batch_size):
    batch_ids = ids[i : i + batch_size]
    batch_docs = all_chunks[i : i + batch_size]
    batch_embeddings = embeddings[i : i + batch_size]
    batch_metadatas = all_metadatas[i : i + batch_size]
    
    print(f"Upserting batch {i} to {i + len(batch_ids)}...")
    collection.upsert(
        ids=batch_ids,
        documents=batch_docs,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )

print(f"Done! {len(all_chunks)} chunks stored in ChromaDB.")
