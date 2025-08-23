
import os
import tempfile
import shutil
import logging
import re
from typing import List, Dict, Optional, Union, Tuple
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
# LlamaIndex imports
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
import chromadb
from llama_index.core.indices.keyword_table.simple_base import SimpleKeywordTableIndex
from llama_index.llms.google_genai import GoogleGenAI 
from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

from llama_index.core.llms import MockLLM # 

# Local imports
import constants as constants
from utils import (
    format_citations_enhanced,
    detect_language,
    validate_input,
    clean_arabic_text,
    create_comprehensive_prompt_v2,
    HybridRetriever,
    MetadataEnhancer,
    ResponseVerifier,
    analyze_answer_completeness
)

# Configure Logging
logging.basicConfig(level=getattr(logging, constants.LOG_LEVEL, "INFO"))
logger = logging.getLogger(__name__)
# to ensure the Chroma DB folder exists on startup
if not os.path.exists(constants.VECTORSTORE_DIR):
    os.makedirs(constants.VECTORSTORE_DIR, exist_ok=True)
    logger.info(f"Created Chroma DB directory at {constants.VECTORSTORE_DIR}")

# Streamlit App Configuration
st.set_page_config(
    page_title=constants.PAGE_CONFIG["page_title"],
    page_icon=constants.PAGE_CONFIG["page_icon"],
    layout=constants.PAGE_CONFIG["layout"],
    initial_sidebar_state=constants.PAGE_CONFIG["initial_sidebar_state"]
)
st.title("🤖 Enhanced Agentic RAG System")
st.caption("Advanced PDF analysis with hybrid search, improved citations, and robust retrieval")

# Environment Variables Check
gemini_api_key = constants.GEMINI_API_KEY
llama_cloud_api_key = constants.LLAMA_CLOUD_API_KEY
if not gemini_api_key:
    st.error(constants.ERROR_MESSAGES["english"]["llm_config_error"])
    st.stop()

# Initialize  LLM with better parameters using the new class
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

# Settings.llm = llm
Settings.embed_model = constants.EMBEDDING_MODEL
Settings.chunk_size = constants.CHUNK_SIZE
Settings.chunk_overlap = constants.CHUNK_OVERLAP

#  Session State
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'indices' not in st.session_state:
    st.session_state.indices = {}
if 'hybrid_retrievers' not in st.session_state:
    st.session_state.hybrid_retrievers = {}
if 'metadata_stats' not in st.session_state:
    st.session_state.metadata_stats = {}

#  Sidebar with Advanced Configuration
with st.sidebar:
    st.header("🔧 Configuration")
    # Document Upload and Processing
    st.subheader("Document Management")
    uploaded_files = st.file_uploader("Upload PDF documents", type=['pdf'],
                                    accept_multiple_files=True)
    if st.button("🚀 Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing documents with  pipeline..."):
                try:
                    Settings.llm = MockLLM() 
                    logger.info("Settings.llm configured to MockLLM for indexing.")

                    # Create temporary directory
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_input_path = os.path.join(tmp_dir, "input")
                        os.makedirs(tmp_input_path, exist_ok=True)
                        # Save uploaded files with metadata 
                        metadata_enhancer = MetadataEnhancer()
                        for uploaded_file in uploaded_files:
                            if uploaded_file.name.lower().endswith('.pdf'):
                                file_path = os.path.join(tmp_input_path, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getvalue())
                                # Extract and store file metadata
                                file_metadata = metadata_enhancer.extract_file_metadata(
                                    uploaded_file.name, uploaded_file.size
                                )
                                st.session_state.metadata_stats[uploaded_file.name] = file_metadata
                        #  Document Loading
                        documents = []
                        if constants.ENABLE_LLAMA_PARSE and llama_cloud_api_key:
                            try:
                                from llama_parse import LlamaParse
                                parser = LlamaParse(
                                    result_type="markdown",
                                    api_key=llama_cloud_api_key,
                                    parsing_instruction=constants.PARSING_INSTRUCTIONS
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
                            #  SimpleDirectoryReader with custom metadata
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
                            # Clear previous data
                            st.session_state.indices = {}
                            st.session_state.hybrid_retrievers = {}
                            st.session_state.chat_history = []
                            #  Node Processing Pipeline
                            if constants.CHUNKING_STRATEGY == "semantic":
                                node_parser = SemanticSplitterNodeParser.from_defaults(
                                    buffer_size=1,
                                    breakpoint_percentile_threshold=95,
                                    embed_model=Settings.embed_model
                                )
                            elif constants.CHUNKING_STRATEGY == "sentence":
                                node_parser = SentenceSplitter(
                                    chunk_size=constants.CHUNK_SIZE,
                                    chunk_overlap=constants.CHUNK_OVERLAP,
                                    paragraph_separator="\n",
                                    secondary_chunking_regex="[.!?]\\s+"
                                )
                            else:  # fixed
                                node_parser = SentenceSplitter(
                                    chunk_size=constants.CHUNK_SIZE,
                                    chunk_overlap=constants.CHUNK_OVERLAP
                                )
                            # Create  ingestion pipeline
                            pipeline = IngestionPipeline(
                                transformations=[
                                    node_parser,
                                    Settings.embed_model
                                ]
                            )
                            if constants.ENABLE_LANGUAGE_SEPARATION:
                                # Process by language
                                language_docs = {"arabic": [], "english": [], "other": []}
                                for doc in documents:
                                    clean_text = clean_arabic_text(doc.text[:1000])
                                    doc_language = detect_language(clean_text)
                                    if doc_language not in language_docs:
                                        doc_language = "other"
                                    language_docs[doc_language].append(doc)
                                # Create indices for each language
                                for lang, docs in language_docs.items():
                                    if docs:
                                        # Process nodes through pipeline
                                        nodes = pipeline.run(documents=docs)
                                        #  metadata for nodes
                                        for node in nodes:
                                            node.metadata.update({
                                                'language': lang,
                                                'chunk_id': node.node_id,
                                                'processing_timestamp': str(os.path.getmtime(tmp_input_path))
                                            })
                                        # Create vector index
                                        os.makedirs(constants.VECTORSTORE_DIR, exist_ok=True)
                                        os.chmod(constants.VECTORSTORE_DIR, 0o755)

                                        collection_name = constants.CHROMA_COLLECTIONS.get(lang, f"legal_documents_{lang}")
                                        chroma_client = chromadb.PersistentClient(path=constants.VECTORSTORE_DIR)
                                        chroma_collection = chroma_client.get_or_create_collection(collection_name)
                                        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                                        storage_context = StorageContext.from_defaults(vector_store=vector_store)
                                        vector_index = VectorStoreIndex(
                                            nodes,
                                            storage_context=storage_context,
                                            show_progress=True
                                        )
                                        # Create keyword index for hybrid search
                                        keyword_index = SimpleKeywordTableIndex(nodes, show_progress=True) # <-- This now uses MockLLM
                                        # Create hybrid retriever
                                        hybrid_retriever = HybridRetriever(
                                            vector_index=vector_index,
                                            keyword_index=keyword_index,
                                            vector_top_k=constants.VECTOR_TOP_K,
                                            keyword_top_k=constants.KEYWORD_TOP_K,
                                            alpha=0.7  # Weight for vector search
                                        )
                                        st.session_state.indices[lang] = {
                                            'vector': vector_index,
                                            'keyword': keyword_index
                                        }
                                        st.session_state.hybrid_retrievers[lang] = hybrid_retriever
                                        st.success(f"✅ Built {lang} indices with {len(docs)} documents")
                            else:
                                # Single unified index
                                nodes = pipeline.run(documents=documents)
                                #  metadata for nodes
                                for node in nodes:
                                    node.metadata.update({
                                        'language': 'unified',
                                        'chunk_id': node.node_id,
                                        'processing_timestamp': str(os.path.getmtime(tmp_input_path))
                                    })
                                # Create unified indices
                                os.makedirs(constants.VECTORSTORE_DIR, exist_ok=True)
                                os.chmod(constants.VECTORSTORE_DIR, 0o755)

                                collection_name = constants.CHROMA_COLLECTIONS["unified"]
                                chroma_client = chromadb.PersistentClient(path=constants.VECTORSTORE_DIR)
                                chroma_collection = chroma_client.get_or_create_collection(collection_name)
                                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                                vector_index = VectorStoreIndex(
                                    nodes,
                                    storage_context=storage_context,
                                    show_progress=True
                                )
                                keyword_index = KeywordTableIndex(nodes, show_progress=True) # <-- This now uses MockLLM
                                hybrid_retriever = HybridRetriever(
                                    vector_index=vector_index,
                                    keyword_index=keyword_index,
                                    vector_top_k=constants.VECTOR_TOP_K,
                                    keyword_top_k=constants.KEYWORD_TOP_K,
                                    alpha=0.7
                                )
                                st.session_state.indices["unified"] = {
                                    'vector': vector_index,
                                    'keyword': keyword_index
                                }
                                st.session_state.hybrid_retrievers["unified"] = hybrid_retriever
                                st.success(f"✅ Built unified indices with {len(documents)} documents")
                            st.success("🎉 Enhanced processing completed!")
                            # Display processing statistics
                            st.info(f"""
                            📊 **Processing Stats:**
                            - Chunking: {constants.CHUNKING_STRATEGY}
                            - Vector Top-K: {constants.VECTOR_TOP_K}
                            - Keyword Top-K: {constants.KEYWORD_TOP_K}
                            - Hybrid Search: {'✅' if constants.ENABLE_HYBRID_SEARCH else '❌'}
                            - Reranking: {'✅' if constants.ENABLE_RERANKING else '❌'}
                            """)
                except Exception as e:
                    logger.error(f"Error processing documents: {e}", exc_info=True)
                    st.error(f"Processing error: {str(e)}")
    st.divider()
    # Metadata Inspection
    if st.session_state.metadata_stats:
        st.subheader("📋 Document Metadata")
        for filename, metadata in st.session_state.metadata_stats.items():
            with st.expander(f"📄 {filename}"):
                st.json(metadata)
    # Clear data button
    if st.button("🗑️ Clear All Data"):
        try:
            if os.path.exists(constants.VECTORSTORE_DIR):
                shutil.rmtree(constants.VECTORSTORE_DIR)
            st.session_state.indices = {}
            st.session_state.hybrid_retrievers = {}
            st.session_state.chat_history = []
            st.session_state.metadata_stats = {}
            st.success("✅ All data cleared successfully!")
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            st.error(f"Error clearing data: {str(e)}")

# Main Chat Interface
if not st.session_state.hybrid_retrievers and not st.session_state.indices:
    st.info("📥 Please upload and process PDF documents to start chatting.")
else:
    #  Query Input
    if prompt := st.chat_input("Ask a question about the documents..."):
        if not validate_input(prompt):
            st.error(constants.ERROR_MESSAGES["english"]["invalid_input"])
        else:
            # Detect query language and select appropriate retriever
            query_language = detect_language(prompt)
            selected_key = "unified"
            if constants.ENABLE_LANGUAGE_SEPARATION and query_language in st.session_state.hybrid_retrievers:
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
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            # Generate  response
            # Replace the query processing section in your main app with this enhanced version
# Enhanced and safe query processing section for your app.py

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    # Use hybrid retriever if available
                    if constants.ENABLE_HYBRID_SEARCH and selected_key in st.session_state.hybrid_retrievers:
                        retriever = st.session_state.hybrid_retrievers[selected_key]
                        retrieved_nodes = retriever.retrieve(prompt)
                    else:
                        # Fallback to vector retriever
                        vector_index = st.session_state.indices[selected_key]['vector']
                        retriever = VectorIndexRetriever(
                            index=vector_index,
                            similarity_top_k=constants.VECTOR_TOP_K
                        )
                        retrieved_nodes = retriever.retrieve(prompt)

                    # Debug logging
                    logger.info(f"Retrieved {len(retrieved_nodes)} nodes for query: {prompt[:50]}...")

                    # Apply reranking if enabled
                    original_nodes_count = len(retrieved_nodes)
                    if constants.ENABLE_RERANKING and retrieved_nodes:
                        if constants.RERANKER_TYPE == "llm":
                            reranker = LLMRerank(
                                llm=llm,
                                top_n=constants.FINAL_TOP_K
                            )
                        else:
                            reranker = SentenceTransformerRerank(
                                model=constants.RERANKER_MODEL,
                                top_n=constants.FINAL_TOP_K
                            )
                        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, QueryBundle(prompt))
                        logger.info(f"After reranking: {len(retrieved_nodes)} nodes (was {original_nodes_count})")

                    # Ensure we have nodes to work with
                    if not retrieved_nodes:
                        st.error("❌ لم يتم العثور على محتوى ذي صلة في الوثائق المرفوعة.")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "❌ لم يتم العثور على محتوى ذي صلة في الوثائق المرفوعة.",
                            "citations": [],
                            "language": query_language,
                            "error": True
                        })
                        

                    # Create enhanced context
                    context_str = "\n".join([
                        f"[Document: {node.metadata.get('file_name', 'Unknown')} | "
                        f"Page: {node.metadata.get('page_label', 'N/A')}]\n{node.get_content()}"
                        for node in retrieved_nodes
                    ])
                
                    # Generate response
                    enhanced_prompt = create_comprehensive_prompt_v2(query_language, context_str, prompt)
                    response_text = llm.complete(enhanced_prompt).text

                    # Display response
                    message_placeholder.markdown(response_text)
                    
                    # Format citations with multiple fallback strategies
                    citations = []
                    try:
                        citations = format_citations_enhanced(retrieved_nodes)
                        logger.info(f"Enhanced citations generated: {len(citations)}")
                    except Exception as e:
                        logger.error(f"Enhanced citation formatting failed: {e}")
                        try:
                            # Fallback to safe citation formatting
                            citations = format_citations_safe_fallback(retrieved_nodes)
                            logger.info(f"Fallback citations generated: {len(citations)}")
                        except Exception as e2:
                            logger.error(f"Fallback citation formatting also failed: {e2}")
                            # Last resort: create basic citations
                            citations = [f"📄 مصدر 1 | {len(retrieved_nodes)} مقاطع مسترجعة"]

                    # Verify response quality (optional, non-blocking)
                    verification_result = {"is_valid": True, "issues": "None"}
                    try:
                        if constants.ENABLE_RESPONSE_VERIFICATION:
                            verifier = ResponseVerifier()
                            verification_result = verifier.verify_response(
                                response_text, prompt, retrieved_nodes, context_str
                            )
                    except Exception as e:
                        logger.warning(f"Response verification failed: {e}")

                    # Always display citations if we have retrieved nodes
                    if citations:
                        st.markdown("**📚 المصادر:**")
                        for i, citation in enumerate(citations, 1):
                            st.markdown(f"{i}. {citation}")
                    else:
                        # This should never happen now, but just in case
                        st.markdown("**📚 المصادر:**")
                        st.markdown(f"1. 📄 وثائق متعددة | {len(retrieved_nodes)} مقطع مسترجع")
                        citations = [f"📄 وثائق متعددة | {len(retrieved_nodes)} مقطع مسترجع"]
                    
                    # Debug information (if enabled)
                    if constants.DEBUG_MODE:
                        with st.expander("🔍 تحليل الجودة"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("عدد المقاطع المسترجعة", len(retrieved_nodes))
                            with col2:
                                st.metric("عدد المصادر المعروضة", len(citations))
                            with col3:
                                avg_score = 0.0
                                try:
                                    scores = [getattr(node, 'score', 0.0) for node in retrieved_nodes if hasattr(node, 'score')]
                                    avg_score = sum(float(s) for s in scores if s is not None) / max(len(scores), 1)
                                except Exception:
                                    avg_score = 0.0
                                st.metric("متوسط نقاط الصلة", f"{avg_score:.2f}")
                            
                            # Show detailed node information
                            if st.checkbox("إظهار تفاصيل المقاطع المسترجعة"):
                                for i, node in enumerate(retrieved_nodes[:5]):  # Show first 5
                                    with st.expander(f"مقطع {i+1}"):
                                        try:
                                            st.write(f"**الملف:** {node.metadata.get('file_name', 'غير محدد')}")
                                            st.write(f"**النقاط:** {getattr(node, 'score', 'غير متاح')}")
                                            content = node.get_content()[:200] + "..." if len(node.get_content()) > 200 else node.get_content()
                                            st.write(f"**المحتوى:** {content}")
                                        except Exception as e:
                                            st.write(f"خطأ في عرض المقطع: {e}")

                    # Store enhanced chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "citations": citations,
                        "language": query_language,
                        "retrieval_method": "hybrid" if constants.ENABLE_HYBRID_SEARCH else "vector",
                        "verification": verification_result,
                        "num_sources": len(retrieved_nodes),
                        "effective_sources": len(citations),
                        "processing_success": True
                    })
                    
                except Exception as e:
                    logger.error(f"Error generating response: {e}", exc_info=True)
                    error_msg = f"❌ خطأ في معالجة الاستفسار: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    
                    # Even in error case, try to show something useful
                    try:
                        if 'retrieved_nodes' in locals() and retrieved_nodes:
                            st.markdown("**📚 المصادر (جزئية):**")
                            st.markdown("1. 📄 مصدر متاح | خطأ في المعالجة")
                            error_citations = ["📄 مصدر متاح | خطأ في المعالجة"]
                        else:
                            error_citations = []
                    except Exception:
                        error_citations = []
                        
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "citations": error_citations,
                        "language": query_language,
                        "error": True
                    })
#  Chat History Display
if st.session_state.chat_history:
    st.subheader("💬  Conversation History")
    for i, message_dict in enumerate(st.session_state.chat_history):
        role = message_dict["role"]
        content = message_dict["content"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)
                #  metadata display
                col1, col2 = st.columns(2)
                with col1:
                    citations = message_dict.get("citations", [])
                    if citations:
                        with st.expander(f"📚 Sources ({len(citations)})"):
                            for j, citation in enumerate(citations, 1):
                                st.markdown(f"{j}. {citation}")
                with col2:
                    if constants.DEBUG_MODE:
                        # Display technical metadata
                        metadata = {
                            "Language": message_dict.get("language", "unknown"),
                            "Retrieval": message_dict.get("retrieval_method", "unknown"),
                            "Sources": message_dict.get("num_sources", 0),
                            "Quality": "✅ Good" if message_dict.get("verification", {}).get("is_valid", True) else "⚠️ Issues"
                        }
                        with st.expander("🔍 Technical Details"):
                            for key, value in metadata.items():
                                st.write(f"**{key}:** {value}")
                            if message_dict.get("verification", {}).get("issues"):
                                st.write(f"**Issues:** {message_dict['verification']['issues']}")

# Sidebar analytics
with st.sidebar:
    if st.session_state.chat_history:
        if constants.DEBUG_MODE:
            st.subheader("📊 Session Analytics")
            total_queries = len([m for m in st.session_state.chat_history if m["role"] == "user"])
            total_responses = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
            avg_sources = sum([m.get("num_sources", 0) for m in st.session_state.chat_history if m["role"] == "assistant"]) / max(total_responses, 1)
            st.metric("Total Queries", total_queries)
            st.metric("Total Responses", total_responses)
            st.metric("Avg Sources/Response", f"{avg_sources:.1f}")
            # Language distribution
            languages = [m.get("language", "unknown") for m in st.session_state.chat_history if m["role"] == "user"]
            if languages:
                lang_counts = {lang: languages.count(lang) for lang in set(languages)}
                st.write("**Language Distribution:**")
                for lang, count in lang_counts.items():
                    st.write(f"- {lang.title()}: {count}")