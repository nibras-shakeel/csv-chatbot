import torch
from langchain_community.llms import CTransformers
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import time
import sys

# Verify GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

DB_FAISS_PATH = r"C:\Users\shake\OneDrive\Desktop\Chat_with_CSV_File_Lllama2\vectorstore\db_faiss"
loader = CSVLoader(file_path=r"C:\Users\shake\OneDrive\Desktop\Chat_with_CSV_File_Lllama2\data\dfProteinPowder.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()

# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Converting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)
docsearch.save_local(DB_FAISS_PATH)

# Use the correct model path
llm = CTransformers(model=r"C:\Users\shake\OneDrive\Desktop\Chat_with_CSV_File_Lllama2\models\llama-2-7b-chat.ggmlv3.q8_0.bin",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.1,
                    device=device)  # Ensure the model uses the GPU if available

qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

while True:
    chat_history = []
    query = input("Input Prompt: ")
    if query.lower() == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue

    start_time = time.time()  # Start timing
    result = qa({"question": query, "chat_history": chat_history})
    end_time = time.time()  # End timing
    
    response_time = end_time - start_time  # Calculate response time
    print("Response:", result['answer'])
    print(f"Response time: {response_time:.4f} seconds")
