"""
@author: Dr Yen Fred WOGUEM 

"""


from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pypdf
import os




# List of all .txt and .pdf files

folder_path = './'

txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

documents = []

# Load all .txt files
for txt_file in txt_files:
    txt_path = os.path.join(folder_path, txt_file)
    loader_txt = TextLoader(txt_path, encoding="utf-8")
    documents.extend(loader_txt.load())  

# Load all .pdf files
for pdf_file in pdf_files:
    pdf_path = os.path.join(folder_path, pdf_file)
    loader_pdf = PyPDFLoader(pdf_path)
    documents.extend(loader_pdf.load())  

# Divide documents (.txt and .pdf) into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)


# Search model configuration
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the FAISS index from documents
faiss_index = FAISS.from_documents(docs, embedding)

# Save index for future use
faiss_index.save_local("mon_index_faiss")

# Loading the FAISS index
faiss_index = FAISS.load_local("mon_index_faiss", embeddings=embedding, allow_dangerous_deserialization=True) # FAISS : Facebook AI Similarity Search

retrieved_docs = faiss_index.similarity_search("What are the latest regulatory updates on insurance?", k=3)

for doc in retrieved_docs:
    print(doc.page_content)
    print(f"Document:\n{doc.page_content[:500]}...")

  

# Loading a Hugging Face pipeline
generator = pipeline("text-generation", model="gpt2", device_map="auto", max_length=500, truncation=True)

# Create an LLM LangChain
llm = HuggingFacePipeline(pipeline=generator)

# RAG generator configuration
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=faiss_index.as_retriever()
)

# Querying the document database
query = "What are the latest regulatory updates on insurance?"
response = qa_chain.invoke(query)

print("**********************************************************************")
print("**********************************************************************")
print("**********************************************************************")
print(response)
print("**********************************************************************")
print("**********************************************************************")
print("**********************************************************************")

