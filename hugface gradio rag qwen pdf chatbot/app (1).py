import gradio as gr
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA

# Global variables
vectorstore = None
qa_chain = None


# 📄 Step 1: Process PDF
def process_pdf(file):
    global vectorstore, qa_chain

    if file is None:
        return "⚠️ Please upload a PDF first."

    # ✅ FIX: Save file properly (HF Spaces issue fix)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file)
        temp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Load Qwen model (lightweight)
    model_name = "Qwen/Qwen1.5-1.8B-Chat"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.7
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return "✅ PDF processed successfully! Now ask your questions."


# 🤖 Step 2: Ask Question
def ask_question(query):
    global qa_chain

    if qa_chain is None:
        return "⚠️ Please upload and process a PDF first."

    if query.strip() == "":
        return "⚠️ Please enter a question."

    response = qa_chain.run(query)
    return response


# 🎨 Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📄 RAG PDF Chatbot (Qwen)")
    gr.Markdown("Upload a PDF and ask questions using AI (RAG + Qwen).")

    with gr.Row():
        file_input = gr.File(
            label="Upload PDF",
            file_types=[".pdf"],
            type="binary"  # ✅ IMPORTANT FIX
        )
        status = gr.Textbox(label="Status")

    process_btn = gr.Button("Process PDF")
    process_btn.click(process_pdf, inputs=file_input, outputs=status)

    gr.Markdown("## 💬 Ask Questions")

    query = gr.Textbox(label="Enter your question")
    answer = gr.Textbox(label="Answer")

    ask_btn = gr.Button("Ask")
    ask_btn.click(ask_question, inputs=query, outputs=answer)

# Launch
demo.launch()