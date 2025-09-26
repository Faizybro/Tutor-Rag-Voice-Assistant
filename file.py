import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document

# Load dataset
with open("quest_and_ans.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Convert JSON to Documents (RAG-friendly)
docs = []
for level, qas in qa_data.items():
    for question, answer in qas.items():
        content = f"Question: {question}\nAnswer: {answer}"
        docs.append(Document(
            page_content=content,
            metadata={"level": level, "question": question}
        ))

# Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS index
db = FAISS.from_documents(docs, embedding_model)
db.save_local("rag_index")

print("✅ FAISS index saved as rag_index")