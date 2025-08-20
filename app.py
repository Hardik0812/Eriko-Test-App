import gradio as gr
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader

def load_doc(list_file_path):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=64
    )
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

def create_db(splits):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)
    return vectordb

def initialize_chatbot(vector_db):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vector_db.as_retriever()
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACE_API_TOKEN"),
        temperature=0.5,
        max_new_tokens=512,
        task="text-generation"  # Explicitly specify the task type
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )
    return qa_chain

def process_and_initialize(files):
    if not files:
        return None, None, "Please upload a file first."
    
    try:
        list_file_path = [file.name for file in files if file is not None]
        doc_splits = load_doc(list_file_path)
        db = create_db(doc_splits)
        qa = initialize_chatbot(db)
        return db, qa, "Database created! Ready for questions."
    except Exception as e:
        return None, None, f"Processing error: {str(e)}"

def user_query_typing_effect(query, qa_chain, chatbot):
    history = chatbot or []
    try:
        response = qa_chain.invoke({"question": query, "chat_history": []})
        assistant_response = response["answer"]
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": ""})
        for i in range(len(assistant_response)):
            history[-1]["content"] += assistant_response[i]
            yield history, ""
            time.sleep(0.03)
    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        yield history, ""

def demo():
    custom_css = """
    body {
        background-color: #FF8C00;
        font-family: Arial, sans-serif;
    }
    .gradio-container {
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.3);
        padding: 20px;
    }
    footer {
        visibility: hidden;
    }
    .chatbot {
        border: 2px solid #000;
        border-radius: 10px;
        background-color: #FFF5E1;
    }
    """
    with gr.Blocks(css=custom_css) as app:
        vector_db = gr.State()
        qa_chain = gr.State()
        gr.Markdown("### 🌟 **PDF & TXT Chatbot Eriko AI** 🌟")
        gr.Markdown("#### Upload your document and ask questions interactively!")
        with gr.Row():
            with gr.Column(scale=1):
                txt_file = gr.Files(
                    label="📁 Upload Documents",
                    file_types=[".txt", ".pdf"],
                    type="filepath"
                )
                analyze_btn = gr.Button("🚀 Process Documents")
                status = gr.Textbox(
                    label="📊 Status",
                    placeholder="Status updates will appear here...",
                    interactive=False
                )
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="🤖 Chat with your data",
                    height=600,
                    bubble_full_width=False,
                    show_label=False,
                    render_markdown=True,
                    type="messages",
                    elem_classes=["chatbot"]
                )
                query_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="Ask about the document...",
                    show_label=False,
                    container=False
                )
                query_btn = gr.Button("Ask")
        analyze_btn.click(
            fn=process_and_initialize,
            inputs=[txt_file],
            outputs=[vector_db, qa_chain, status],
            show_progress="minimal"
        )
        query_btn.click(
            fn=user_query_typing_effect,
            inputs=[query_input, qa_chain, chatbot],
            outputs=[chatbot, query_input],
            show_progress="minimal"
        )
        query_input.submit(
            fn=user_query_typing_effect,
            inputs=[query_input, qa_chain, chatbot],
            outputs=[chatbot, query_input],
            show_progress="minimal"
        )
    app.launch()

if __name__ == "__main__":
    app = demo()
    port = int(os.environ.get("PORT", "7860"))
    print(f"Binding Gradio on 0.0.0.0:{port}", flush=True)
    # queue() helps avoid blocking during model downloads on small instances
    app.queue().launch(server_name="0.0.0.0", server_port=port, show_api=False)
