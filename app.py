from typing import List, Dict, Optional, Union, Tuple
from dotenv import load_dotenv
import logging
import streamlit as st
import constants
import tempfile
import os
import shutil
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
    Response,
    ServiceContext,
    KeywordTableIndex,
    ComposableGraph,
    QueryBundle
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank, LLMRerank
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
    QueryFusionRetriever
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.keyword_table.simple_base import SimpleKeywordTableIndex

# Use the recommended Google GenAI integration
from llama_index.llms.google_genai import GoogleGenAI

from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.llms import MockLLM

from utils import (
    format_citations_enhanced,
    detect_language,
    validate_input,
    clean_arabic_text,
    create_comprehensive_prompt_v2,
    HybridRetriever,
    MetadataEnhancer,
    ResponseVerifier
)
load_dotenv()

logging.basicConfig(level=getattr(logging, constants.LOG_LEVEL, "INFO"))
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=constants.PAGE_CONFIG["page_title"],
    page_icon=constants.PAGE_CONFIG["page_icon"],
    layout=constants.PAGE_CONFIG["layout"],
    initial_sidebar_state=constants.PAGE_CONFIG["initial_sidebar_state"]
)
st.title("Agentic RAG System")
st.caption(" PDF analysis with hybrid search, improved citations, and robust retrieval")

gemini_api_key = constants.GEMINI_API_KEY
if not gemini_api_key:
    st.error(constants.ERROR_MESSAGES["english"]["llm_config_error"])
    st.stop()

try:
    llm = GoogleGenAI(
        model=constants.LLM_MODEL_NAME,
        api_key=gemini_api_key,
        temperature=0.0,
        max_tokens=4096,
    )
    logger.info(f"LLM initialized: {constants.LLM_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    st.error(f"{constants.ERROR_MESSAGES['english']['llm_config_error']}: {str(e)}")
    st.stop()

Settings.embed_model = constants.EMBEDDING_MODEL
Settings.chunk_size = constants.CHUNK_SIZE
Settings.chunk_overlap = constants.CHUNK_OVERLAP

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'indices' not in st.session_state:
    st.session_state.indices = {}
if 'hybrid_retrievers' not in st.session_state:
    st.session_state.hybrid_retrievers = {}
if 'metadata_stats' not in st.session_state:
    st.session_state.metadata_stats = {}

with st.sidebar:
    st.header("Configuration")
    st.subheader("Document Management")
    use_llamaparse = st.checkbox("Use LlamaParse for document parsing", value=False)
    uploaded_files = st.file_uploader("Upload PDF documents", type=['pdf'], accept_multiple_files=True)
    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing documents with pipeline..."):
                try:
                    Settings.llm = MockLLM()
                    logger.info("Settings.llm configured to MockLLM for indexing.")

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_input_path = os.path.join(tmp_dir, "input")
                        os.makedirs(tmp_input_path, exist_ok=True)
                        metadata_enhancer = MetadataEnhancer()
                        for uploaded_file in uploaded_files:
                            if uploaded_file.name.lower().endswith('.pdf'):
                                file_path = os.path.join(tmp_input_path, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getvalue())
                                file_metadata = metadata_enhancer.extract_file_metadata(
                                    uploaded_file.name, uploaded_file.size
                                )
                                st.session_state.metadata_stats[uploaded_file.name] = file_metadata
                        documents = []
                        if use_llamaparse:
                            # Use LlamaParse API for document parsing (direct integration)
                            llama_parse_api_key = constants.LLAMAPARSE_API_KEY
                            if not llama_parse_api_key:
                                st.error("LlamaParse API key missing. Please set LLAMAPARSE_API_KEY in your .env file.")
                                st.stop()
                            try:
                                from llama_parse import LlamaParse
                                parser = LlamaParse(
                                    result_type="markdown",
                                    api_key=llama_parse_api_key,
                                    parsing_instruction=getattr(constants, "PARSING_INSTRUCTIONS", None)
                                )
                                for uploaded_file in uploaded_files:
                                    if uploaded_file.name.lower().endswith('.pdf'):
                                        temp_file_path = os.path.join(tmp_input_path, uploaded_file.name)
                                        parsed_docs = parser.load_data(temp_file_path)
                                        # Enhanced metadata for each document
                                        for doc in parsed_docs:
                                            doc.metadata.update({
                                                'file_name': uploaded_file.name,
                                                'filename': uploaded_file.name,
                                                'file_path': temp_file_path,
                                                'file_size': uploaded_file.size,
                                                'processing_method': 'llama_parse'
                                            })
                                        documents.extend(parsed_docs)
                            except Exception as e:
                                logger.error(f"LlamaParse error: {e}")
                                st.error(f"LlamaParse error: {str(e)}")
                                st.stop()
                        else:
                            # Use default SimpleDirectoryReader
                            reader = SimpleDirectoryReader(
                                input_dir=tmp_input_path,
                                filename_as_id=True,
                                file_metadata=lambda file_path: {
                                    'file_name': os.path.basename(file_path),
                                    'filename': os.path.basename(file_path),
                                    'file_path': file_path,
                                    'processing_method': 'simple_reader'
                                }
                            )
                            documents = reader.load_data()
                        if not documents:
                            st.error("No documents could be loaded.")
                        else:
                            st.session_state.indices = {}
                            st.session_state.hybrid_retrievers = {}
                            st.session_state.chat_history = []
                            node_parser = SentenceSplitter(
                                chunk_size=constants.CHUNK_SIZE,
                                chunk_overlap=constants.CHUNK_OVERLAP
                            )
                            pipeline = IngestionPipeline(
                                transformations=[
                                    node_parser,
                                    Settings.embed_model
                                ]
                            )
                            language_docs = {"arabic": [], "english": [], "other": []}
                            for doc in documents:
                                clean_text = clean_arabic_text(doc.text[:1000])
                                doc_language = detect_language(clean_text)
                                if doc_language not in language_docs:
                                    doc_language = "other"
                                language_docs[doc_language].append(doc)
                            for lang, docs in language_docs.items():
                                if docs:
                                    nodes = pipeline.run(documents=docs)
                                    for node in nodes:
                                        node.metadata.update({
                                            'language': lang,
                                            'chunk_id': node.node_id,
                                            'processing_timestamp': str(os.path.getmtime(tmp_input_path))
                                        })
                                    # Get or create ChromaDB collection for language
                                    collection_name = constants.CHROMA_COLLECTIONS.get(lang, f"legal_documents_{lang}")
                                    chroma_client = chromadb.PersistentClient(path=constants.VECTORSTORE_DIR)
                                    chroma_collection = chroma_client.get_or_create_collection(collection_name)
                                    # Set up vector store and storage context
                                    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                                    # Build vector index for semantic search
                                    vector_index = VectorStoreIndex(
                                        nodes,
                                        storage_context=storage_context,
                                        show_progress=True
                                    )
                                    # Build keyword index for keyword-based search
                                    keyword_index = SimpleKeywordTableIndex(nodes, show_progress=True)
                                    # Create hybrid retriever combining both search methods
                                    hybrid_retriever = HybridRetriever(
                                        vector_index=vector_index,
                                        keyword_index=keyword_index,
                                        vector_top_k=constants.VECTOR_TOP_K,
                                        keyword_top_k=constants.KEYWORD_TOP_K,
                                        alpha=0.7
                                    )
                                    # Store indices and retriever in session state
                                    st.session_state.indices[lang] = {
                                        'vector': vector_index,
                                        'keyword': keyword_index
                                    }
                                    st.session_state.hybrid_retrievers[lang] = hybrid_retriever
                                    st.success(f"Built {lang} indices with {len(docs)} documents")
                            st.success("processing completed!")
                except Exception as e:
                    logger.error(f"Error processing documents: {e}", exc_info=True)
                    st.error(f"Processing error: {str(e)}")
    st.divider()
    if st.session_state.metadata_stats:
        st.subheader("Document Metadata")
        for filename, metadata in st.session_state.metadata_stats.items():
            with st.expander(f"{filename}"):
                st.json(metadata)
    if st.button("Clear All Data"):
        try:
            if os.path.exists(constants.VECTORSTORE_DIR):
                shutil.rmtree(constants.VECTORSTORE_DIR)
            st.session_state.indices = {}
            st.session_state.hybrid_retrievers = {}
            st.session_state.chat_history = []
            st.session_state.metadata_stats = {}
            st.success("All data cleared successfully!")
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            st.error(f"Error clearing data: {str(e)}")

if not st.session_state.hybrid_retrievers and not st.session_state.indices:
    st.info("Please upload and process PDF documents to start chatting.")
else:
    if prompt := st.chat_input("Ask a question about the documents..."):
        if not validate_input(prompt):
            st.error(constants.ERROR_MESSAGES["english"]["invalid_input"])
        else:
            query_language = detect_language(prompt)
            selected_key = "unified"
            if query_language in st.session_state.hybrid_retrievers:
                selected_key = query_language
            elif "unified" in st.session_state.hybrid_retrievers:
                selected_key = "unified"
            else:
                available_keys = list(st.session_state.hybrid_retrievers.keys())
                if available_keys:
                    selected_key = available_keys[0]
                else:
                    st.error("No retriever available.")
                    st.stop()
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    retriever = st.session_state.hybrid_retrievers[selected_key]
                    retrieved_nodes = retriever.retrieve(prompt)
                    if retrieved_nodes:
                        reranker = SentenceTransformerRerank(
                            model=constants.RERANKER_MODEL,
                            top_n=constants.FINAL_TOP_K
                        )
                        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, QueryBundle(prompt))
                    context_str = "\n".join([
                        f"[Document: {node.metadata.get('file_name', 'Unknown')} | "
                        f"Page: {node.metadata.get('page_label', 'N/A')}]\n{node.get_content()}"
                        for node in retrieved_nodes
                    ])
                    enhanced_prompt = create_comprehensive_prompt_v2(query_language, context_str, prompt)
                    response_text = llm.complete(enhanced_prompt).text
                    verifier = ResponseVerifier()
                    verification_result = verifier.verify_response(
                        response_text, prompt, retrieved_nodes, context_str
                    )
                    citations = format_citations_enhanced(retrieved_nodes)
                    message_placeholder.markdown(response_text)
                    if citations:
                        st.markdown("**Sources:**")
                        for i, citation in enumerate(citations, 1):
                            st.markdown(f"{i}. {citation}")
                    if not verification_result['is_valid']:
                        st.warning(f"Response Quality Issues: {verification_result['issues']}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "citations": citations,
                        "language": query_language,
                        "retrieval_method": "hybrid",
                        "verification": verification_result,
                        "num_sources": len(retrieved_nodes)
                    })
                except Exception as e:
                    logger.error(f"Error generating response: {e}", exc_info=True)
                    error_msg = f"Query processing error: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "citations": [],
                        "language": query_language,
                        "error": True
                    })

if st.session_state.chat_history:
    st.subheader("Conversation History")
    for i, message_dict in enumerate(st.session_state.chat_history):
        role = message_dict["role"]
        content = message_dict["content"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)
                col1, col2 = st.columns(2)
                with col1:
                    citations = message_dict.get("citations", [])
                    if citations:
                        with st.expander(f"Sources ({len(citations)})"):
                            for citation in citations:
                                st.markdown(citation)
                with col2:
                    if os.getenv("DEBUG_MODE", "False") == "True":
                        metadata = {
                            "Language": message_dict.get("language", "unknown"),
                            "Retrieval": message_dict.get("retrieval_method", "unknown"),
                            "Sources": message_dict.get("num_sources", 0),
                            "Quality": "Good" if message_dict.get("verification", {}).get("is_valid", True) else "Issues"
                        }
                        with st.expander("Technical Details"):
                            for key, value in metadata.items():
                                st.markdown(f"{key}: {value}")
                            if message_dict.get("verification", {}).get("issues"):
                                st.markdown(f"Issues: {message_dict['verification']['issues']}")

