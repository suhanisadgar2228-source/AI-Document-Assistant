
import streamlit as st

from modules.document_loader import load_document
from modules.text_splitter import split_documents
from modules.vector_store import VectorStore
from modules.qa_engine import answer_question


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------


# ================= CUSTOM UI STYLING =================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #f7f5ff;
    color: #29213d;
}

/* Main content */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5def5;
}

[data-testid="stSidebar"] * {
    color: #30234b;
}

/* Main headings */
h1, h2, h3 {
    color: #512b81 !important;
    font-weight: 700 !important;
}

/* Normal text */
.stMarkdown p,
.stMarkdown li,
.stCaption {
    color: #403951;
}

/* Buttons */
.stButton > button {
    background-color: #7042a6;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #512b81;
    color: white !important;
    border: none;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 1px dashed #9b79c8;
    border-radius: 12px;
    padding: 12px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #ffffff;
    border: 1px solid #e8e0f2;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #ffffff;
    border: 1px solid #b69ad9;
    border-radius: 12px;
}

/* Text input */
.stTextInput input,
.stTextArea textarea {
    background-color: #ffffff;
    color: #29213d;
    border: 1px solid #cbb9e2;
    border-radius: 8px;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #e5def5;
    border-radius: 10px;
}

/* Success messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #e5def5;
}

</style>
""", unsafe_allow_html=True)

# ================= END CUSTOM UI STYLING =================



# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------


st.markdown("""
<style>

/* Force readable text throughout the application */
.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp div,
.stApp small {
    color: #302344;
}

/* Main background */
.stApp {
    background-color: #f7f5ff;
}

/* Main content area */
.main .block-container {
    background-color: #f7f5ff;
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Headings */
h1, h2, h3, h4 {
    color: #512b81 !important;
    font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
}

[data-testid="stSidebar"] * {
    color: #302344 !important;
}

/* Normal markdown content */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong {
    color: #302344 !important;
}

/* Buttons */
.stButton > button {
    background-color: #7042a6 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton > button * {
    color: #ffffff !important;
}

.stButton > button:hover {
    background-color: #512b81 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border: 1px solid #cbb9e2 !important;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] * {
    color: #302344 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 1px solid #e5def5 !important;
    border-radius: 14px !important;
}

[data-testid="stChatMessage"] * {
    color: #302344 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border: 1px solid #9b79c8 !important;
}

[data-testid="stChatInput"] textarea {
    color: #302344 !important;
    background-color: #ffffff !important;
    -webkit-text-fill-color: #302344 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #756b82 !important;
    -webkit-text-fill-color: #756b82 !important;
}

/* Other input fields */
.stTextInput input,
.stTextArea textarea {
    color: #302344 !important;
    background-color: #ffffff !important;
    -webkit-text-fill-color: #302344 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e5def5 !important;
}

[data-testid="stExpander"] * {
    color: #302344 !important;
}

/* Captions */
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #655a75 !important;
}

/* Dividers */
hr {
    border-color: #e5def5 !important;
}

</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# --------------------------------------------------
# HELPER FUNCTION: DISPLAY SOURCES
# --------------------------------------------------

def display_sources(sources):
    """Display retrieved document sources."""

    if not sources:
        return

    with st.expander("📚 View sources used", expanded=False):

        for number, source in enumerate(sources, start=1):

            filename = source.get("source", "Unknown document")
            page = source.get("page")
            score = source.get("score")

            location = (
                f"Page {page}"
                if page is not None
                else "Document"
            )

            st.markdown(
                f"**Source {number}: {filename}** · {location}"
            )

            if score is not None:
                st.caption(f"Similarity score: {score:.3f}")

            preview = source.get("text", "").strip()

            if preview:
                st.caption(
                    preview[:700] +
                    ("..." if len(preview) > 700 else "")
                )

            if number < len(sources):
                st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 📚 AI Assistant")
    st.caption("Your intelligent document workspace")

    st.divider()

    st.markdown("### 📁 Upload documents")

    uploads = st.file_uploader(
        "Choose PDF, DOCX or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple documents.",
    )

    if uploads:
        st.caption(f"Selected files: {len(uploads)}")

    process_clicked = st.button(
        "⚡ Process documents",
        type="primary",
        use_container_width=True,
    )

    if process_clicked:

        if not uploads:
            st.warning("Please upload at least one document.")

        else:
            try:

                with st.spinner("Reading uploaded documents..."):

                    documents = []

                    for upload in uploads:
                        documents.extend(
                            load_document(upload)
                        )

                with st.spinner("Splitting text into chunks..."):

                    chunks = split_documents(documents)

                if not chunks:

                    st.error(
                        "No readable text was found in the uploaded files."
                    )

                else:

                    with st.spinner(
                        "Generating embeddings and building search index..."
                    ):

                        store = VectorStore()
                        store.build(chunks)

                    st.session_state.vector_store = store

                    st.session_state.processed_files = [
                        file.name for file in uploads
                    ]

                    st.session_state.chunk_count = len(chunks)
                    st.session_state.chat_history = []

                    st.success(
                        f"Successfully processed {len(uploads)} "
                        f"document(s) and created {len(chunks)} chunks."
                    )

            except Exception as exc:

                st.error(f"Processing failed: {exc}")

    st.divider()

    st.markdown("### 📄 Processed documents")

    if st.session_state.processed_files:

        for filename in st.session_state.processed_files:
            st.markdown(f"📄 {filename}")

    else:
        st.caption("No documents processed yet.")

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.chat_history = []
        st.rerun()

    if st.button(
        "Reset document index",
        use_container_width=True,
    ):

        st.session_state.vector_store = None
        st.session_state.processed_files = []
        st.session_state.chat_history = []
        st.session_state.chunk_count = 0

        st.rerun()

    st.divider()

    st.caption("Powered by Python, FAISS and Groq AI")


# --------------------------------------------------
# MAIN DASHBOARD HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">AI Document Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Upload documents, understand information and get intelligent answers.'
    '</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero-card">
    <h2>✨ Your documents. Your AI assistant.</h2>
    <p>
        Ask questions, summarize documents and discover important
        information using an AI-powered document search system.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DASHBOARD STATISTICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">📄 Processed documents</div>
        <div class="stat-value">
            {len(st.session_state.processed_files)}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">🔍 Searchable chunks</div>
        <div class="stat-value">
            {st.session_state.chunk_count}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">💬 Chat messages</div>
        <div class="stat-value">
            {len(st.session_state.chat_history)}
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if st.session_state.vector_store is None:

    st.markdown(
        '<div class="section-heading">🚀 Get started</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">
            <h3>📤 Upload your documents</h3>
            <p>
                Upload PDF, Word or text files using the sidebar.
                You can process multiple documents together.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>💡 Ask intelligent questions</h3>
            <p>
                Ask questions about your documents and receive
                answers based on the uploaded content.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">
            <h3>🔎 Smart document search</h3>
            <p>
                Semantic search retrieves relevant passages
                using embeddings and the FAISS vector database.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>🤖 AI-powered answers</h3>
            <p>
                Groq language models generate clear answers
                with supporting document references.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "To begin, upload documents in the left sidebar "
        "and click Process documents."
    )


else:

    st.markdown(
        '<div class="section-heading">💬 Chat with your documents</div>',
        unsafe_allow_html=True,
    )

    # Display previous messages

    for message_number, message in enumerate(
        st.session_state.chat_history
    ):

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):
                display_sources(message["sources"])

            if (
                message["role"] == "assistant"
                and message.get("content")
            ):

                st.download_button(
                    label="⬇️ Download this answer",
                    data=message["content"],
                    file_name=f"AI_answer_{message_number}.txt",
                    mime="text/plain",
                    key=f"download_{message_number}",
                )

    # New question input

    question = st.chat_input(
        "Ask anything about your uploaded documents..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "🔎 Searching documents and generating your answer..."
                ):

                    answer, sources = answer_question(
                        question,
                        st.session_state.vector_store,
                    )

                st.markdown(answer)

                display_sources(sources)

                st.download_button(
                    label="⬇️ Download answer",
                    data=answer,
                    file_name="AI_Document_Answer.txt",
                    mime="text/plain",
                    key=f"new_answer_{len(st.session_state.chat_history)}",
                )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as exc:

                error_text = (
                    f"Sorry, I couldn't generate an answer: {exc}"
                )

                st.error(error_text)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                })


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer-text">
    AI Document Assistant · Built with Python, Streamlit,
    FAISS, Sentence Transformers and Groq AI
</div>
""", unsafe_allow_html=True)
