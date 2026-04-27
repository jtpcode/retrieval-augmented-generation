import re
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Safely delete the collection if it already exists, for example if you updated the text files
# and run this script again.
try:
    chroma_client.delete_collection(name="superstore_sales")
except ValueError:
    pass  # The collection doesn't exist yet in the first run

collection = chroma_client.get_or_create_collection(name="superstore_sales")

MONTHS = (
    "January|February|March|April|May|June|"
    "July|August|September|October|November|December"
)


def extract_metadata_from_chunk(chunk, source):
    meta = {"source": source}

    # Year: all unique years in the chunk, sorted
    years = sorted(set(re.findall(r'\b(201[4-7])\b', chunk)))
    if years:
        meta["year"] = ", ".join(years)

    # Month: all unique months in order of appearance
    months = list(dict.fromkeys(re.findall(MONTHS, chunk)))
    if months:
        meta["month"] = ", ".join(months)

    # Category: all unique categories in order of appearance
    categories = list(dict.fromkeys(re.findall(r'\b(Furniture|Office Supplies|Technology)\b', chunk, re.IGNORECASE)))
    if categories:
        meta["category"] = ", ".join(categories)

    # Sub-category: all unique sub-categories in order of appearance
    sub_cats = list(dict.fromkeys(re.findall(r'subcategory ([\w][\w ]+?)[),]', chunk)))
    if sub_cats:
        meta["sub_category"] = ", ".join(sub_cats)

    # Region: all unique regions in order of appearance
    regions = list(dict.fromkeys(
        r.capitalize() for r in re.findall(r'\b(Central|East|South|West)\b region', chunk, re.IGNORECASE)
    ))
    if regions:
        meta["region"] = ", ".join(regions)

    return meta



CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

print("Processing text files with langchain...")
text_files_dir = Path("text_files")
all_chunks = []
all_metadatas = []

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# Process each text file, split into chunks, and extract metadata
for file in text_files_dir.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    chunks = splitter.split_text(text)
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk:
            all_chunks.append(chunk)
            all_metadatas.append(extract_metadata_from_chunk(chunk, file.name))

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

# NOTE: ChromaDB has a batch size limit of 5461, which is relevant with small chunk size.
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
