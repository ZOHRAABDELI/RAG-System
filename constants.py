import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# --- API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Optional: LlamaParse API key for advanced document parsing
LLAMA_CLOUD_API_KEY = os.getenv("LLAMAPARSE_API_KEY")

# ---  Model Configuration ---
LLM_MODEL_NAME = "models/gemini-2.0-flash"  

#  Embedding Model - BGE-M3 with optimized settings
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_MODEL = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    max_length=512,  
    normalize=True   # Normalize embeddings for better similarity
)

# Alternative embedding models for specific use cases
EMBEDDING_MODELS = {
    "multilingual": "BAAI/bge-m3",
    "english": "BAAI/bge-large-en-v1.5", 
    "arabic": "aubmindlab/bert-base-arabertv02",
    "legal": "nlpaueb/legal-bert-base-uncased"
}

# --- Enhanced Vector Store Configuration ---
VECTORSTORE_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "enhanced_legal_documents"

# Language-specific and domain-specific collections
CHROMA_COLLECTIONS = {
    "arabic": "legal_documents_ar_v2",
    "english": "legal_documents_en_v2",
    "other": "legal_documents_other_v2",
    "unified": "legal_documents_unified_v2",
    "contracts": "legal_contracts",
    "regulations": "legal_regulations",
    "policies": "legal_policies"
}

# ---  Document Processing Configuration ---
# Optimized chunking parameters 
CHUNK_SIZE = 1200           
CHUNK_OVERLAP = 400        
MIN_CHUNK_SIZE = 100      
MAX_CHUNK_SIZE = 2000   

# Semantic chunking parameters
SEMANTIC_SIMILARITY_THRESHOLD = 0.8
SEMANTIC_BUFFER_SIZE = 2

# ---  Retrieval Configuration ---

DEFAULT_SIMILARITY_TOP_K = 70   # Increased from 60    
VECTOR_TOP_K = 70              # Increased from 60 - Casts a wider vector net
KEYWORD_TOP_K = 50             # Increased from 40 - Casts a wider keyword net
FINAL_TOP_K = 35               # Increased from 25 - CRITICAL: More context to LLM

# Similarity thresholds 
SIMILARITY_THRESHOLD = 0.4    
HIGH_CONFIDENCE_THRESHOLD = 0.8  
LOW_CONFIDENCE_THRESHOLD = 0.3   
# ---  Reranking Configuration ---
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TYPE = "sentence-transformer"
RERANKER_MODELS = {
    "fast": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "balanced": "cross-encoder/ms-marco-MiniLM-L-6-v2", 
    "high_quality": "cross-encoder/ms-marco-electra-base"
}

# Reranking parameters
RERANK_TOP_N = 20               # Number of results to rerank
RERANK_BATCH_SIZE = 32          # Batch size for reranking
RERANK_SCORE_THRESHOLD = 0.1    # Minimum rerank score

# --- Language Detection Configuration ---
ARABIC_THRESHOLD = 0.25         # Lowered for better detection
MIN_TEXT_LENGTH = 5             # Minimum text length for detection
LANGUAGE_CONFIDENCE_THRESHOLD = 0.8

# Enhanced language patterns
ARABIC_SCRIPT_RANGES = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
]

# --- Enhanced Legal Patterns ---
ARABIC_LEGAL_PATTERNS = {
    'articles': [
        r'المادة\s+\d+',
        r'المادة\s+رقم\s+\d+', 
        r'م\.\s*\d+',
        r'مادة\s+\d+',
        r'الفصل\s+\d+'
    ],
    'paragraphs': [
        r'الفقرة\s+\d+',
        r'فقرة\s+\d+', 
        r'ف\.\s*\d+',
        r'البند\s+\d+'
    ],
    'sections': [
        r'القسم\s+\d+',
        r'الباب\s+\d+', 
        r'ق\.\s*\d+',
        r'الجزء\s+\d+'
    ],
    'chapters': [
        r'الباب\s+\d+',
        r'الفصل\s+\d+', 
        r'ب\.\s*\d+',
        r'الكتاب\s+\d+'
    ],
    'clauses': [
        r'البند\s+\d+',
        r'الشرط\s+\d+',
        r'النقطة\s+\d+'
    ]
}

ENGLISH_LEGAL_PATTERNS = {
    'articles': [
        r'Article\s+\d+',
        r'Art\.\s*\d+', 
        r'A\.\s*\d+',
        r'Section\s+\d+',
        r'§\s*\d+'
    ],
    'paragraphs': [
        r'Paragraph\s+\d+',
        r'Para\.\s*\d+', 
        r'P\.\s*\d+',
        r'\(\d+\)',
        r'Subsection\s+\d+'
    ],
    'sections': [
        r'Section\s+\d+',
        r'Sec\.\s*\d+', 
        r'S\.\s*\d+',
        r'Part\s+\d+'
    ],
    'chapters': [
        r'Chapter\s+\d+',
        r'Ch\.\s*\d+', 
        r'C\.\s*\d+',
        r'Title\s+\d+'
    ],
    'clauses': [
        r'Clause\s+\d+',
        r'Cl\.\s*\d+',
        r'Item\s+\d+',
        r'Point\s+\d+'
    ]
}

# --- Enhanced Response Templates ---
ARABIC_RESPONSE_TEMPLATE = """
**الإجابة المفصلة:**
{response_content}

**المصادر والمراجع:**
{sources}

**معلومات إضافية:**
- عدد المصادر المستخدمة: {num_sources}
- مستوى الثقة: {confidence_level}
- طريقة البحث: {search_method}
"""

ENGLISH_RESPONSE_TEMPLATE = """
**Detailed Answer:**
{response_content}

**Sources and References:**
{sources}

**Additional Information:**
- Number of sources used: {num_sources}
- Confidence level: {confidence_level}
- Search method: {search_method}
"""

# --- Security and Validation Configuration ---
MAX_QUERY_LENGTH = 3000          
MAX_RESPONSE_LENGTH = 8000      
ALLOWED_FILE_TYPES = ['.pdf', '.docx', '.txt', '.md']
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file
MAX_TOTAL_SIZE = 500 * 1024 * 1024  # 500MB total

# Enhanced security patterns
SUSPICIOUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',
    r'javascript\s*:',
    r'data\s*:\s*text/html',
    r'eval\s*\(',
    r'exec\s*\(',
    r'<iframe[^>]*>',
    r'on\w+\s*=',
    r'document\.',
    r'window\.',
    r'\.innerHTML',
    r'\.outerHTML'
]

# Content validation patterns
VALID_QUERY_PATTERNS = [
    r'what|who|where|when|why|how',  # Question words
    r'list|show|explain|define|describe',  # Action words
    r'[\u0600-\u06FF]+',  # Arabic text
    r'[a-zA-Z]+',  # English text
]

# ---  Logging Configuration ---
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
LOG_FILE = "enhanced_rag_agent.log"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# --- Performance Configuration ---
ENABLE_CACHING = True
CACHE_SIZE = 1000
CACHE_TTL = 3600  # 1 hour

# Processing timeouts
DOCUMENT_PROCESSING_TIMEOUT = 300  # 5 minutes
QUERY_PROCESSING_TIMEOUT = 60      # 1 minute
EMBEDDING_TIMEOUT = 120            # 2 minutes

# --- Feature Flags ---
ENABLE_RERANKING = True
ENABLE_LANGUAGE_SEPARATION = False
ENABLE_HYBRID_SEARCH = True
ENABLE_SEMANTIC_CHUNKING = False
ENABLE_RESPONSE_VERIFICATION = False
ENABLE_METADATA_ENHANCEMENT = True
ENABLE_QUERY_OPTIMIZATION = True
ENABLE_LLAMA_PARSE = False
CHUNKING_STRATEGY = "fixed" # Options: fixed, sentence, semantic

# Advanced features
ENABLE_QUERY_EXPANSION = False      # Experimental
ENABLE_ANSWER_FUSION = False        # Experimental
ENABLE_FACT_CHECKING = False        # Experimental

# --- UI Configuration ---
PAGE_CONFIG = {
    "page_title": "Agentic RAG System",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

STREAMLIT_THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#ffffff",
    "secondaryBackgroundColor": "#f0f2f6",
    "textColor": "#262730",
    "font": "sans serif"
}

# Chart and visualization settings
CHART_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e", 
    "success": "#2ca02c",
    "warning": "#ff9800",
    "error": "#d62728"
}
# --- LlamaParse Configuration ---
PARSING_INSTRUCTIONS = """
This document contains legal text that may include:
- Articles, sections, and subsections with numbering
- Lists of  principles, rights, or obligations
- Complex  terminology and cross-references
- Tables and structured information

Please preserve:
- All numbering systems (articles, sections, paragraphs)
- List structures and bullet points
- Legal terminology exactly as written
- Cross-references between sections
- Table structures where possible

Focus on maintaining the hierarchical structure and legal precision of the document.
"""
# ---  Error Messages ---
ERROR_MESSAGES = {
    "arabic": {
        "no_documents": "لم يتم العثور على وثائق. يرجى رفع ملفات PDF أولاً.",
        "processing_error": "حدث خطأ أثناء معالجة الوثائق.",
        "query_error": "حدث خطأ أثناء معالجة استفسارك.",
        "invalid_input": "الاستفسار غير صالح. يرجى إعادة صياغة السؤال.",
        "no_answer": "لا توجد إجابة في الوثائق المقدمة.",
        "llm_config_error": "خطأ في تكوين LLM. تحقق من مفتاح API والنقطة النهائية.",
        "llama_parse_key_missing": "مفتاح API الخاص بـ LlamaParse مفقود. يرجى إضافته إلى ملف .env.",
        "embedding_error": "خطأ في نموذج التضمين. تحقق من التكوين.",
        "retrieval_error": "خطأ في عملية البحث والاسترجاع.",
        "reranking_error": "خطأ في عملية إعادة الترتيب.",
        "timeout_error": "انتهت مهلة المعالجة. يرجى المحاولة مرة أخرى.",
        "file_too_large": "حجم الملف كبير جداً. الحد الأقصى المسموح: {max_size}MB.",
        "unsupported_format": "تنسيق الملف غير مدعوم. الأنواع المدعومة: {formats}.",
        "metadata_extraction_error": "خطأ في استخراج البيانات الوصفية.",
        "language_detection_error": "خطأ في تحديد اللغة.",
        "chunking_error": "خطأ في تقسيم النص إلى أجزاء.",
        "indexing_error": "خطأ في فهرسة الوثائق.",
        "verification_failed": "فشل في التحقق من جودة الاستجابة."
    },
    "english": {
        "no_documents": "No documents found. Please upload files first.",
        "processing_error": "Error occurred while processing documents.",
        "query_error": "Error occurred while processing your query.",
        "invalid_input": "Invalid query. Please rephrase your question.",
        "no_answer": "No answer found in the provided documents.",
        "llm_config_error": "LLM configuration error. Check API key and endpoint.",
        "llama_parse_key_missing": "LlamaParse API key missing. Please add it to your .env file.",
        "embedding_error": "Embedding model error. Check configuration.",
        "retrieval_error": "Error in search and retrieval process.",
        "reranking_error": "Error in reranking process.",
        "timeout_error": "Processing timeout. Please try again.",
        "file_too_large": "File too large. Maximum allowed size: {max_size}MB.",
        "unsupported_format": "Unsupported file format. Supported types: {formats}.",
        "metadata_extraction_error": "Error extracting metadata.",
        "language_detection_error": "Error detecting language.",
        "chunking_error": "Error chunking text.",
        "indexing_error": "Error indexing documents.",
        "verification_failed": "Response quality verification failed."
    }
}

# ---  Success Messages ---
SUCCESS_MESSAGES = {
    "arabic": {
        "documents_processed": "تم معالجة الوثائق بنجاح!",
        "index_built": "تم بناء الفهرس بنجاح!",
        "data_cleared": "تم مسح البيانات بنجاح.",
        "embedding_complete": "تم إنشاء التضمينات بنجاح.",
        "retrieval_complete": "تم البحث والاسترجاع بنجاح.",
        "reranking_complete": "تم إعادة الترتيب بنجاح.",
        "response_generated": "تم إنشاء الاستجابة بنجاح.",
        "metadata_extracted": "تم استخراج البيانات الوصفية بنجاح.",
        "language_detected": "تم تحديد اللغة بنجاح.",
        "chunks_created": "تم تقسيم النص بنجاح.",
        "verification_passed": "تم التحقق من جودة الاستجابة بنجاح."
    },
    "english": {
        "documents_processed": "Documents processed successfully!",
        "index_built": "Index built successfully!",
        "data_cleared": "Data cleared successfully.",
        "embedding_complete": "Embeddings created successfully.",
        "retrieval_complete": "Search and retrieval completed successfully.",
        "reranking_complete": "Reranking completed successfully.",
        "response_generated": "Response generated successfully.",
        "metadata_extracted": "Metadata extracted successfully.",
        "language_detected": "Language detected successfully.",
        "chunks_created": "Text chunking completed successfully.",
        "verification_passed": "Response quality verification passed."
    }
}



# --- Quality Assurance Configuration ---
QUALITY_METRICS = {
    "response_length_min": 20,
    "response_length_max": 5000,
    "citation_requirement": True,
    "source_verification": True,
    "language_consistency": True,
    "factual_accuracy": True
}

# Response quality thresholds
QUALITY_THRESHOLDS = {
    "excellent": 0.9,
    "good": 0.7,
    "acceptable": 0.5,
    "poor": 0.3
}

# ---  Retrieval Strategies ---
RETRIEVAL_STRATEGIES = {
    "precision": {
        "vector_top_k": 20,
        "keyword_top_k": 10,
        "final_top_k": 8,
        "similarity_threshold": 0.7,
        "alpha": 0.8  # Favor vector search
    },
    "recall": {
        "vector_top_k": 50,
        "keyword_top_k": 25,
        "final_top_k": 20,
        "similarity_threshold": 0.4,
        "alpha": 0.6  # Balanced approach
    },
    "balanced": {
        "vector_top_k": 30,
        "keyword_top_k": 15,
        "final_top_k": 12,
        "similarity_threshold": 0.5,
        "alpha": 0.7  # Slight favor to vector
    }
}

# --- Document Type Specific Configuration ---
DOCUMENT_TYPES = {
    "law": {
        "chunking_strategy": "semantic",
        "chunk_size": 1500,
        "overlap": 500,
        "embedding_model": "legal"
    },
    "contract": {
        "chunking_strategy": "sentence",
        "chunk_size": 1000,
        "overlap": 300,
        "embedding_model": "multilingual"
    },
    "regulation": {
        "chunking_strategy": "fixed",
        "chunk_size": 1200,
        "overlap": 400,
        "embedding_model": "multilingual"
    },
    "policy": {
        "chunking_strategy": "semantic",
        "chunk_size": 800,
        "overlap": 200,
        "embedding_model": "multilingual"
    }
}

# --- Monitoring and Analytics ---
MONITORING_CONFIG = {
    "track_queries": True,
    "track_responses": True,
    "track_performance": True,
    "track_errors": True,
    "analytics_retention_days": 30
}

# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    "document_processing_time": 60,  # seconds per MB
    "query_response_time": 10,       # seconds
    "embedding_time": 5,             # seconds per chunk
    "retrieval_time": 3              # seconds
}

# --- Export Configuration ---
EXPORT_FORMATS = {
    "json": True,
    "csv": True,
    "pdf": False,  # Requires additional dependencies
    "xlsx": True
}

# --- Development and Debug Configuration ---
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "False").lower() == "true"
PROFILING_ENABLED = os.getenv("PROFILING_ENABLED", "False").lower() == "true"

# Debug settings
if DEBUG_MODE:
    LOG_LEVEL = "DEBUG"
    CHUNK_SIZE = 800 
    VECTOR_TOP_K = 20  

# --- Version and Compatibility ---
RAG_SYSTEM_VERSION = "2.0.0"
SUPPORTED_LLAMAINDEX_VERSION = ">=0.10.0"
SUPPORTED_PYTHON_VERSION = ">=3.8"

# ---  Configuration ---
BEST_PRACTICES = {
    "chunk_overlap_ratio": 0.3,  
    "max_chunks_per_doc": 1000,
    "min_chunk_chars": 100,
    "max_chunk_chars": 2000,
    "rerank_threshold": 0.1,
    "response_verification": True,
    "metadata_enrichment": True,
    "query_preprocessing": True,
    "response_postprocessing": True
}