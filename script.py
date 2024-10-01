
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
import torch
# Verify GPU availability if not then use CPU: 
# NOTE FOR PROFESSOR: In order to use with a GPU please install the CUDA framework

device = 'GPU' if torch.cuda.is_available() else 'CPU'
print(f"Using device: {device}")
DB_FAISS_PATH = r"C:\Users\shake\OneDrive\Desktop\OD31bot\vectorstore\db_faiss"  # Please replace this with the file path on your computer
loader = CSVLoader(file_path=r"C:\Users\shake\OneDrive\Desktop\OD31bot\data\dfProteinPowder3.csv", encoding="utf-8", csv_args={'delimiter': ','})# use dfProteinPowder3 for best performance, the other two run out of tokens sometimes
data = loader.load()
print(data)
# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
print(len(text_chunks))
# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# Converting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)
docsearch.save_local(DB_FAISS_PATH)
llm = CTransformers(model=r"C:\Users\shake\OneDrive\Desktop\OD31bot\models\llama-2-7b-chat.ggmlv3.q8_0.bin", #Replace this too with the correct file path from your computer (The llama2 Inside the model folder)
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.1)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())
while True:
    chat_history = []
    query = input("Input Prompt: ")
    if query.lower() == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa({"question": query, "chat_history": chat_history})
    print("Response:", result['answer'])
