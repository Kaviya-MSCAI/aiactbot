import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "aiactbot"
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return SentenceTransformerEmbeddings(model_name=EMBED_MODEL)


def load_and_index_pdf(pdf_path: str, progress_callback=None):
    if progress_callback:
        progress_callback("📄 Loading PDF...")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    if progress_callback:
        progress_callback(f"✂️ Splitting {len(pages)} pages into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)

    if progress_callback:
        progress_callback(f"🧠 Embedding {len(chunks)} chunks — this takes a minute...")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )

    if progress_callback:
        progress_callback(f"✅ Done! {len(chunks)} chunks indexed.")

    return len(chunks), vectorstore


def load_existing_vectorstore():
    if not os.path.exists(CHROMA_PATH):
        return None
    try:
        embeddings = get_embeddings()
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        count = vectorstore._collection.count()
        if count == 0:
            return None
        return vectorstore
    except Exception:
        return None


def get_chunk_count():
    vs = load_existing_vectorstore()
    if vs is None:
        return 0
    return vs._collection.count()


def build_chain(vectorstore):
    llm = OllamaLLM(model="llama3.2", temperature=0.1)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_template("""
You are AIActBot — a plain English EU AI Act compliance assistant.
You help companies and individuals in Ireland understand the EU AI Act and GDPR.

Your rules:
- Always answer in plain simple English
- Cite the specific Article or Section your answer comes from
- If the answer is not in the documents say "I couldn't find this in the loaded documents"
- Mention practical implications for Irish and EU companies when relevant

Context from documents:
{context}

Question: {question}

Answer in plain English with article references:""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_question(chain_tuple, question: str):
    chain, retriever = chain_tuple

    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    sources = []
    seen = set()
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "Document")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "page": page + 1 if isinstance(page, int) else page,
                "source": os.path.basename(source),
                "snippet": doc.page_content[:200].strip()
            })

    return {
        "answer": answer,
        "sources": sources
    }