import chromadb

# 1. Connect to your existing local database
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Get the collection you created
collection = chroma_client.get_collection(name="superstore_sales")

# 3. See how many items are in the database
total_items = collection.count()
print(f"Total documents in database: {total_items}")

# 4. "Peek" at the first 2 records (returns ids, documents, metadatas)
print("\n--- Peeking at the first 2 records ---")
results = collection.peek(limit=2)

for i in range(len(results['ids'])):
    print(f"ID: {results['ids'][i]}")
    print(f"Document: {results['documents'][i]}")
    print(f"Metadata: {results['metadatas'][i]}")
    print("-" * 50)

# 5. Example of viewing specific metadata using .get()
# This gets up to 3 documents where the region is "Central"
print("\n--- Querying by Metadata (Region: Central) ---")
central_results = collection.get(
    where={"region": "Central"},
    limit=3
)

for i in range(len(central_results['ids'])):
    print(f"Document: {central_results['documents'][i]}")
    print(f"Metadata: {central_results['metadatas'][i]}")
    print("-" * 50)