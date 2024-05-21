import chromadb
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Initialize the ChromaDB HTTP client
client = chromadb.HttpClient(host="localhost", port=3000)


# Create a collection for a user in ChromaDB
def create_user_collection(user_id):
    collection_name = f"user_{user_id}_collection"
    client.create_collection(collection_name)
    return collection_name


# Extract metadata from a PDF file
def extract_metadata_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        metadata = reader.metadata
        print(metadata)
        return metadata


# Extract text content from a PDF file using PDFMinerLoader
def extract_text_from_pdf(pdf_path):
    loader = PDFMinerLoader(pdf_path)
    documents = loader.load()
    text = " ".join(doc.page_content for doc in documents)
    return text


# Generate embeddings for text chunks extracted from a PDF
def generate_embeddings(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings_model = SentenceTransformerEmbeddings()
    embeddings = embeddings_model.embed_documents(chunks)
    return chunks, embeddings


# Save the embeddings and metadata to a user's collection in ChromaDB
def save_embeddings(user_id, book_id, pdf_path):
    collection_name = f"user_{user_id}_collection"

    # Check if the collection exists, create if not
    existing_collections = client.list_collections()
    if collection_name not in [collection.name for collection in existing_collections]:
        collection = client.create_collection(collection_name)
    else:
        collection = client.get_collection(collection_name)

    chunks, embeddings = generate_embeddings(pdf_path)
    metadata = extract_metadata_from_pdf(pdf_path)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        document_id = f"{book_id}_{i}"
        collection.add(document_id, embedding, {
            "text": chunk,
            # "metadata": metadata,
            "book_id": book_id,
            "user_id": user_id
        })
    return {f'Successfully saved embeddings and metadata to Chroma for user {user_id} and book {book_id}'}


# Main function to execute the save_embeddings function with example parameters
if __name__ == "__main__":
    save_embeddings(user_id="123456",
                    book_id="book_678901",
                    pdf_path="data/romeo-and-juliet.pdf")
