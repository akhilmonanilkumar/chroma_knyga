# def get_relevant_documents(query, user_id, book_id):
#     query_embedding = embedding_fn.embed_query(query)
#     collection_name = f"user_{user_id}_collection"
#     collection = chroma_client.get_collection(collection_name)
#     results = collection.query(
#         query_embeddings=query_embedding,
#         where={"book_id": book_id},
#         n_results=5)
#     # documents = []
#     # for doc_content, metadata in zip(results["documents"][0], results["metadatas"][0]):
#     #     documents.append(Document(page_content=doc_content, metadata=metadata))
#     documents = []
#     for doc_content, metadata in zip(results["documents"][0], results["metadatas"][0]):
#         document = {
#             "page_content": doc_content,
#             "metadata": metadata
#         }
#         documents.append(document)
#     print(documents)
#     return documents

# Create a collection for a user in ChromaDB
# def create_collection(user_id):
#     collection_name = f"user_{user_id}_collection"
#     chroma_client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
#     return collection_name
#
#
# def get_collections():
#     try:
#         collections = chroma_client.list_collections()
#         print(collections)
#         return collections
#     except Exception as e:
#         print(f"An error occurred: {e}")
#
#
# def delete_collection(user_id):
#     try:
#         collection_name = f"user_{user_id}_collection"
#         chroma_client.delete_collection(collection_name)
#         print("Collection deleted successfully.")
#         return {f"{collection_name} deleted successfully."}
#     except Exception as e:
#         print(f"An error occurred: {e}")
#
#
# def delete_collections():
#     try:
#         collections = get_collections()
#         for collection in collections:
#             print(collection.name)
#             chroma_client.delete_collection(collection.name)
#         print("Collections deleted successfully.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
