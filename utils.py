
import re
import logging
import os
from typing import List, Dict, Optional, Any, Union, Tuple
from langdetect import detect, LangDetectException
from llama_index.core.schema import NodeWithScore
from llama_index.core.response import Response
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle
import hashlib
from datetime import datetime
from collections import defaultdict

import constants as constants

logger = logging.getLogger(__name__)

def format_citations_enhanced(retrieved_nodes: List[NodeWithScore]) -> List[str]:
    """
    Enhanced citation formatting with improved metadata extraction and fallback mechanisms.
    """
    file_to_pages = defaultdict(list)
    seen_citations = set()
    for i, node_with_score in enumerate(retrieved_nodes):
        try:
            # Handle both NodeWithScore and regular nodes
            if isinstance(node_with_score, NodeWithScore):
                node = node_with_score.node
                score = getattr(node_with_score, 'score', None)
            else:
                node = node_with_score
                score = None 

            if score is not None:
                try:
                    score = float(score) 
                except (ValueError, TypeError):
                    score = None 

            if not hasattr(node, 'metadata'):
                logger.warning(f"Node {i} missing metadata")
                continue
            metadata = node.metadata
            file_name = (
                metadata.get('file_name') or
                metadata.get('filename') or
                metadata.get('source') or
                metadata.get('document_id') or
                os.path.basename(metadata.get('file_path', '')) or
                f"Document_{i+1}"
            )
            # Clean and validate file name
            if file_name == '' or file_name == 'Unknown File':
                # Try to extract from file_path or other metadata
                file_path = metadata.get('file_path', '')
                if file_path:
                    file_name = os.path.basename(file_path)
                else:
                    # Generate a meaningful name from content hash or other metadata
                    content_hash = hashlib.md5(node.get_content()[:100].encode()).hexdigest()[:8]
                    file_name = f"Document_{content_hash}"
            #  page extraction with multiple fallbacks
            page_info = (
                metadata.get('page_label') or
                metadata.get('page') or
                metadata.get('page_number') or
                metadata.get('chunk_id', '')
            )
            # Extract section or chapter info if available
            section_info = metadata.get('section', metadata.get('chapter', ''))
            # Build comprehensive page part
            page_parts = []
            if page_info and page_info != 'N/A':
                if str(page_info).isdigit():
                    page_parts.append(f"Page {page_info}")
                else:
                    page_parts.append(f"Section {page_info}")
            if section_info:
                page_parts.append(f"Chapter {section_info}")
            if score is not None and isinstance(score, (int, float)) and score > 0: # <-- Safer check
                page_parts.append(f"(Relevance: {score:.2f})")
            # Add processing method if available
            processing_method = metadata.get('processing_method', '')
            if processing_method:
                page_parts.append(f"[{processing_method}]")
            page_str = " | ".join(page_parts)
            # Create unique identifier for deduplication
            citation_key = (file_name, str(page_info), section_info)
            if citation_key not in seen_citations and page_str:
                file_to_pages[file_name].append(page_str)
                seen_citations.add(citation_key)
        except Exception as e:
            logger.error(f"Error processing citation for node {i}: {e}")
            # Fallback citation
            file_to_pages[f"Document_{i+1}"].append("Error in metadata extraction")

    citations = []
    for file_name, pages in file_to_pages.items():
        if pages:
            citation = f"📄 {file_name} | " + ", ".join(pages)
            citations.append(citation)
    return citations

def detect_language(text: str) -> str:
    """
    Enhanced language detection with better Arabic support and confidence scoring.
    """
    if not text or len(text.strip()) < constants.MIN_TEXT_LENGTH:
        return "unknown"
    
    try:
        # Clean text for better detection
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text.strip())
        
        # Enhanced Arabic character detection
        arabic_patterns = [
            r'[\u0600-\u06FF]',  # Arabic block
            r'[\u0750-\u077F]',  # Arabic Supplement
            r'[\u08A0-\u08FF]',  # Arabic Extended-A
            r'[\uFB50-\uFDFF]',  # Arabic Presentation Forms-A
            r'[\uFE70-\uFEFF]'   # Arabic Presentation Forms-B
        ]
        
        arabic_chars = 0
        for pattern in arabic_patterns:
            arabic_chars += len(re.findall(pattern, text))
        
        total_chars = len(re.sub(r'\s', '', text))
        arabic_ratio = arabic_chars / max(total_chars, 1)
        
        # If significant Arabic content, return Arabic
        if arabic_ratio > constants.ARABIC_THRESHOLD:
            return "arabic"
        
        # Use langdetect for other languages with confidence checking
        try:
            detected = detect(clean_text)
            if detected == 'ar':
                return "arabic"
            elif detected == 'en':
                return "english"
            else:
                return "other"
        except LangDetectException:
            # Fallback to character-based detection
            if arabic_chars > 0:
                return "arabic"
            elif re.search(r'[a-zA-Z]', text):
                return "english"
            else:
                return "unknown"
                
    except Exception as e:
        logger.warning(f"Error in language detection: {e}")
        # Final fallback
        if re.search(r'[\u0600-\u06FF]', text):
            return "arabic"
        elif re.search(r'[a-zA-Z]', text):
            return "english"
        else:
            return "unknown"

def validate_input(query: str) -> bool:
    """
    Enhanced input validation with security checks and content analysis.
    """
    if not query or len(query.strip()) < 2:
        return False
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query.strip())
    
    # Length validation
    if len(query) > constants.MAX_QUERY_LENGTH:
        logger.warning(f"Query too long: {len(query)} characters")
        return False
    
    # Security pattern checks
    for pattern in constants.SUSPICIOUS_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            logger.warning(f"Suspicious pattern detected: {pattern}")
            return False
    
    # Content validation - ensure meaningful content
    words = query.split()
    if len(words) < 1:
        return False
    
    # Check for minimum meaningful characters
    meaningful_chars = len(re.sub(r'[^\w\u0600-\u06FF]', '', query))
    if meaningful_chars < 2:
        return False
    
    return True

def clean_arabic_text(text: str) -> str:
    """
    Enhanced Arabic text cleaning and normalization.
    """
    if not text:
        return text
    
    # Remove diacritics (Tashkeel)
    diacritics_pattern = r'[\u064B-\u065F\u0670\u0640]'
    text = re.sub(diacritics_pattern, '', text)
    
    # Normalize Alef variations
    alef_variations = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا'}
    for variant, normalized in alef_variations.items():
        text = text.replace(variant, normalized)
    
    # Normalize Teh Marbuta
    text = text.replace('ة', 'ه')
    
    # Normalize Yeh variations
    yeh_variations = {'ي': 'ى', 'ئ': 'ى'}
    for variant, normalized in yeh_variations.items():
        text = text.replace(variant, normalized)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def create_comprehensive_prompt_v2(query_language: str, context_str: str, query_str: str) -> str:
    """
    Enhanced prompt creation with better anti-hallucination measures and context handling.
    """
    if query_language == "arabic":
        return f"""أنت مساعد ذكي متخصص في الإجابة على الأسئلة بناءً *حصرياً* على المعلومات المقدمة. اتبع هذه القواعد بدقة شديدة:

**القواعد الأساسية:**
1. **الاعتماد الحصري على السياق:** استخدم فقط المعلومات الموجودة في الوثائق المقدمة أدناه. لا تفترض أو تضيف أي معلومات خارجية.
2. **الدقة في الاستخلاص:** عند طلب قوائم محددة، قدم *جميع* العناصر كما وردت تماماً في النص، مع الإشارة إلى رقم المادة والفقرة.
3. **التحقق من الاكتمال:** ابحث في *كامل* السياق المقدم للتأكد من عدم تفويت أي معلومات ذات صلة.
4. **المصطلحات :** استخدم المصطلحات كما وردت في النص دون تغيير أو استبدال.
5. **الشفافية:** إذا لم تجد معلومات كافية، أجب بوضوح: "لا توجد معلومات كافية في الوثائق المقدمة."
6. **البحث الكامل**: ابحث في *كل* السياق المقدم أدناه - اقرأ كل سطر
7. **عدم التوقف المبكر**: لا تتوقف عند أول نتيجة - استمر حتى نهاية النص
8. **القوائم الشاملة**: إذا طُلبت قائمة، يجب تضمين *جميع* العناصر - صفر استثناءات
9. **الترقيم الأصلي**: احتفظ بأرقام المواد والفقرات كما وردت تماماً
10. **الاستشهاد الإجباري**: لكل نقطة، اذكر (المادة X، الفقرة Y)
**تنبيه هام:** راجع إجابتك قبل الإرسال وتأكد من عدم نسيان أي عنصر موجود في السياق.
**السياق المقدم:**
{context_str}

**السؤال:**
{query_str}

**التعليمات الإضافية:**
- ابدأ مباشرة بالإجابة دون مقدمات
- رقم كل نقطة أو عنصر مع الإشارة إلى مصدرها
- تأكد من الاكتمال والدقة
- لا تضف تفسيرات أو تعليقات إضافية غير موجودة في النص

**الإجابة:**"""

    else:  # English
        return f"""You are an intelligent assistant specialized in  answering questions based *exclusively* on the provided information. Follow these rules with strict precision:

**Core Rules:**
1. **Exclusive Context Reliance:** Use only information explicitly present in the documents provided below. Do not assume or add external information.
2. **Extraction Precision:** When specific lists are requested, provide *all* elements exactly as stated in the text, with proper article and paragraph references.
3. **Completeness Verification:** Search the *entire* provided context to ensure no relevant information is missed.
4. ** Terminology:** Use terms exactly as they appear in the text without alteration or substitution.
5. **Transparency:** If insufficient information is found, clearly respond: "Insufficient information in provided documents."
6. **Complete Search**: Search through *ALL* the context provided below - read every line
7. **No Early Stopping**: Don't stop at the first result - continue to the end of text
8. **Exhaustive Lists**: If a list is requested, include *ALL* elements - zero exceptions
9. **Original Numbering**: Preserve article and paragraph numbers exactly as stated
10. **Mandatory Citations**: For each point, include (Article X, Paragraph Y)
**Important:** Review your answer before sending and ensure no element from context is forgotten.

**Provided Context:**
{context_str}

**Question:**
{query_str}

**Additional Instructions:**
- Start directly with the answer, no preambles
- Number each point or element with source references
- Ensure completeness and accuracy
- Do not add interpretations or comments not found in the text

**Answer:**"""

class HybridRetriever(BaseRetriever):
    """
    Enhanced hybrid retriever combining vector and keyword search with intelligent fusion.
    """
    def __init__(
        self,
        vector_index,
        keyword_index, 
        vector_top_k: int = 20,
        keyword_top_k: int = 10,
        alpha: float = 0.7,
        similarity_threshold: float = 0.0
    ):
        self.vector_index = vector_index
        # Store the keyword index object itself
        self.keyword_index = keyword_index
        self.vector_top_k = vector_top_k
        self.keyword_top_k = keyword_top_k
        self.alpha = alpha  # Weight for vector search
        self.similarity_threshold = similarity_threshold
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Perform hybrid retrieval with intelligent score fusion.
        """
        query_str = query_bundle.query_str
        vector_nodes = []
        keyword_nodes = []

        # Vector retrieval
        try:
            vector_retriever = self.vector_index.as_retriever(
                similarity_top_k=self.vector_top_k
            )
            vector_nodes = vector_retriever.retrieve(query_str)
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")

        # Keyword retrieval
        try:
            # Use the stored keyword index object correctly
            # Ensure the keyword index has an 'as_retriever' method
            if hasattr(self.keyword_index, 'as_retriever'):
                 keyword_retriever = self.keyword_index.as_retriever(similarity_top_k=self.keyword_top_k)
                 keyword_nodes = keyword_retriever.retrieve(query_str)
            else:
                 logger.warning("Keyword index does not support 'as_retriever'. Skipping keyword retrieval.")
        except Exception as e:
             logger.error(f"Error in keyword retrieval: {e}")

        # Create node score maps
        vector_scores = {}
        keyword_scores = {}
        for node_with_score in vector_nodes:
            # Ensure node_with_score is NodeWithScore and has a score
            if isinstance(node_with_score, NodeWithScore) and node_with_score.score is not None:
                try:
                    vector_scores[node_with_score.node.node_id] = float(node_with_score.score)
                except (ValueError, TypeError):
                     vector_scores[node_with_score.node.node_id] = 0.0 # Default to 0 if score invalid
            elif isinstance(node_with_score, NodeWithScore): # Score is None
                 vector_scores[node_with_score.node.node_id] = 0.0 # Treat None score as 0

        for node_with_score in keyword_nodes:
             # Ensure node_with_score is NodeWithScore and has a score
             # Note: KeywordTableSimpleRetriever might not return scores, or return None
             if isinstance(node_with_score, NodeWithScore) and node_with_score.score is not None:
                 try:
                     keyword_scores[node_with_score.node.node_id] = float(node_with_score.score)
                 except (ValueError, TypeError):
                      keyword_scores[node_with_score.node.node_id] = 0.0 # Default to 0 if score invalid
             elif isinstance(node_with_score, NodeWithScore): # Score is None
                  keyword_scores[node_with_score.node.node_id] = 0.0 # Treat None score as 0

        # Get all unique nodes
        all_node_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        node_map = {}
        # Build node map (prefer vector node if exists, otherwise keyword node)
        for node in vector_nodes + keyword_nodes: # Order matters if IDs overlap
            if isinstance(node, NodeWithScore):
                node_map[node.node.node_id] = node.node
            else:
                # Handle case where item might not be NodeWithScore (less likely after retrieval)
                pass

        # Calculate hybrid scores
        hybrid_nodes = []
        for node_id in all_node_ids:
            if node_id not in node_map:
                continue
            # Get scores, defaulting to 0.0 if not found or None
            vector_score = vector_scores.get(node_id, 0.0)
            keyword_score = keyword_scores.get(node_id, 0.0)

            # Ensure scores are floats (they should be after the conversion above, but double-check)
            try:
                vector_score = float(vector_score)
            except (ValueError, TypeError):
                vector_score = 0.0
            try:
                keyword_score = float(keyword_score)
            except (ValueError, TypeError):
                keyword_score = 0.0

            # Normalize scores (assuming max score of 1.0, adjust if needed)
            vector_score_norm = min(vector_score, 1.0)
            keyword_score_norm = min(keyword_score, 1.0) # If keyword score is 0 or None, this is 0

            # Hybrid score calculation
            hybrid_score = (
                self.alpha * vector_score_norm +
                (1 - self.alpha) * keyword_score_norm
            )

            try:
                hybrid_score = float(hybrid_score)
            except (ValueError, TypeError):
                hybrid_score = 0.0 # Default to 0 if calculation failed

            # Apply similarity threshold (ensure self.similarity_threshold is a number)
            try:
                threshold = float(self.similarity_threshold)
            except (ValueError, TypeError):
                threshold = 0.0

            if hybrid_score >= threshold: # <-- This is the '<' comparison that could fail if hybrid_score was None
                hybrid_nodes.append(
                    NodeWithScore(node=node_map[node_id], score=hybrid_score)
                )
        # Sort by hybrid score and return top results
        hybrid_nodes.sort(key=lambda x: (x.score if x.score is not None else -1), reverse=True) # Handle potential None in sorting
        return hybrid_nodes[:max(self.vector_top_k, self.keyword_top_k)]

class MetadataEnhancer:
    """
    Enhanced metadata extraction and enrichment for better document tracking.
    """
    
    def extract_file_metadata(self, filename: str, file_size: int) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from file information.
        """
        file_ext = os.path.splitext(filename)[1].lower()
        
        metadata = {
            'filename': filename,
            'file_name': filename,
            'file_extension': file_ext,
            'file_size': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'processing_timestamp': datetime.now().isoformat(),
            'file_hash': hashlib.md5(filename.encode()).hexdigest()
        }
        
        # Extract potential document type from filename
        doc_type_indicators = {
            'law': ['law', 'legal', 'statute', 'regulation', 'قانون', 'نظام'],
            'contract': ['contract', 'agreement', 'عقد', 'اتفاقية'],
            'policy': ['policy', 'procedure', 'سياسة', 'إجراء'],
            'manual': ['manual', 'guide', 'دليل', 'مرشد']
        }
        
        filename_lower = filename.lower()
        for doc_type, indicators in doc_type_indicators.items():
            if any(indicator in filename_lower for indicator in indicators):
                metadata['document_type'] = doc_type
                break
        else:
            metadata['document_type'] = 'general'
        
        return metadata

class ResponseVerifier:
    """
    Advanced response verification to detect potential hallucinations and quality issues.
    """
    
    def verify_response(
        self, 
        response: str, 
        query: str, 
        retrieved_nodes: List[NodeWithScore],
        context: str
    ) -> Dict[str, Any]:
        """
        Comprehensive response verification.
        """
        issues = []
        is_valid = True
        
        # Check 1: Response relevance to query
        query_terms = set(re.findall(r'\w+', query.lower()))
        response_terms = set(re.findall(r'\w+', response.lower()))
        relevance_score = len(query_terms & response_terms) / max(len(query_terms), 1)
        
        if relevance_score < 0.1:
            issues.append("Low relevance to query")
            is_valid = False
        
        # Check 2: Response grounding in context
        context_terms = set(re.findall(r'\w+', context.lower()))
        grounding_score = len(response_terms & context_terms) / max(len(response_terms), 1)
        
        if grounding_score < 0.3:
            issues.append("Poor grounding in provided context")
            is_valid = False
        
        # Check 3: Length appropriateness
        if len(response.strip()) < 10:
            issues.append("Response too short")
            is_valid = False
        elif len(response) > 5000:
            issues.append("Response potentially too verbose")
        
        # Check 4: Hallucination indicators
        hallucination_patterns = [
            r'i think', r'probably', r'maybe', r'it seems',
            r'أعتقد', r'ربما', r'يبدو', r'من المحتمل'
        ]
        
        for pattern in hallucination_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append("Contains uncertainty indicators")
                break
        
        # Check 5: Source citation verification
        if not retrieved_nodes:
            issues.append("No source nodes available")
            is_valid = False
        
        return {
            'is_valid': is_valid,
            'issues': '; '.join(issues) if issues else 'None',
            'relevance_score': relevance_score,
            'grounding_score': grounding_score,
            'response_length': len(response),
            'num_sources': len(retrieved_nodes)
        }

# Additional utility functions for enhanced RAG

def extract_legal_references(text: str, language: str = "english") -> List[Dict[str, str]]:
    """
    Extract legal references (articles, sections, etc.) from text.
    """
    references = []
    
    if language == "arabic":
        patterns = constants.ARABIC_LEGAL_PATTERNS
    else:
        patterns = constants.ENGLISH_LEGAL_PATTERNS
    
    for ref_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                references.append({
                    'type': ref_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
    
    return references

def calculate_chunk_relevance(chunk_text: str, query: str) -> float:
    """
    Calculate relevance score between chunk and query using multiple methods.
    """
    # Simple term overlap
    chunk_terms = set(re.findall(r'\w+', chunk_text.lower()))
    query_terms = set(re.findall(r'\w+', query.lower()))
    
    if not query_terms:
        return 0.0
    
    overlap_score = len(chunk_terms & query_terms) / len(query_terms)
    
    # Boost score for exact phrase matches
    query_phrases = re.findall(r'\"([^\"]+)\"', query)
    phrase_boost = 0.0
    for phrase in query_phrases:
        if phrase.lower() in chunk_text.lower():
            phrase_boost += 0.2
    
    return min(overlap_score + phrase_boost, 1.0)

def optimize_retrieval_parameters(
    query: str, 
    index_size: int, 
    query_complexity: str = "medium"
) -> Dict[str, int]:
    """
    Dynamically optimize retrieval parameters based on query and index characteristics.
    """
    base_params = {
        "vector_top_k": constants.VECTOR_TOP_K,
        "keyword_top_k": constants.KEYWORD_TOP_K,
        "final_top_k": constants.FINAL_TOP_K
    }
    
    # Adjust based on query length and complexity
    query_length = len(query.split())
    
    if query_length > 20:  # Complex query
        base_params["vector_top_k"] = min(base_params["vector_top_k"] * 1.5, 60)
        base_params["keyword_top_k"] = min(base_params["keyword_top_k"] * 1.3, 25)
    elif query_length < 5:  # Simple query
        base_params["vector_top_k"] = max(base_params["vector_top_k"] * 0.8, 15)
        base_params["keyword_top_k"] = max(base_params["keyword_top_k"] * 0.8, 8)
    
    # Adjust based on index size
    if index_size < 100:  # Small index
        base_params["vector_top_k"] = min(base_params["vector_top_k"], 20)
        base_params["keyword_top_k"] = min(base_params["keyword_top_k"], 10)
    elif index_size > 1000:  # Large index
        base_params["vector_top_k"] = min(base_params["vector_top_k"] * 1.2, 80)
        base_params["keyword_top_k"] = min(base_params["keyword_top_k"] * 1.2, 30)
    
    # Ensure final_top_k is reasonable
    base_params["final_top_k"] = min(
        base_params["final_top_k"],
        max(base_params["vector_top_k"], base_params["keyword_top_k"]) // 2
    )
    
    return {k: int(v) for k, v in base_params.items()}

# Error Messages and Success Messages (maintaining compatibility)
ERROR_MESSAGES = constants.ERROR_MESSAGES
SUCCESS_MESSAGES = constants.SUCCESS_MESSAGES