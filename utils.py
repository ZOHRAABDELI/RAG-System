
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

# --- Place this updated function in enhanced_app.py, replacing the existing one ---
def format_citations_enhanced(retrieved_nodes: List[NodeWithScore]) -> List[str]:
    """
    Enhanced citation formatting that always shows sources when nodes are available,
    but intelligently deduplicates and prioritizes the most relevant ones.
    """
    if not retrieved_nodes:
        return []
    
    # Dictionary to hold the best citation info per file based on highest score
    # Key: file_name, Value: {'page_str': str, 'score': float, 'content': str}
    file_citations = {}
    
    # Collect all scores to determine thresholds
    scores = []
    for node_with_score in retrieved_nodes:
        try:
            if isinstance(node_with_score, NodeWithScore):
                score = getattr(node_with_score, 'score', 0.0)
                try:
                    scores.append(float(score) if score is not None else 0.0)
                except (ValueError, TypeError):
                    scores.append(0.0)
        except Exception:
            continue
    
    # Set a more lenient threshold - we want to show sources, not hide them
    if scores:
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        # Much more lenient threshold - only filter out truly irrelevant content
        MIN_RELEVANCE_FOR_CITATION = max(0.05, mean_score * 0.1)  # Very low threshold
    else:
        MIN_RELEVANCE_FOR_CITATION = 0.0  # Show everything if no scores

    for i, node_with_score in enumerate(retrieved_nodes):
        try:
            # Handle both NodeWithScore and potential fallbacks
            if isinstance(node_with_score, NodeWithScore):
                node = node_with_score.node
                score = getattr(node_with_score, 'score', 0.0)
            else:
                # Fallback - treat as node with default score
                node = node_with_score
                score = 0.0

            # Safely convert score to float
            try:
                score = float(score) if score is not None else 0.0
            except (ValueError, TypeError):
                score = 0.0

            if not hasattr(node, 'metadata'):
                logger.warning(f"Node {i} missing metadata, using defaults")
                # Create minimal metadata for nodes without it
                node.metadata = {'file_name': f'Document_{i+1}'}

            metadata = node.metadata

            # --- Robust File Name Extraction ---
            file_name = (
                metadata.get('file_name') or
                metadata.get('filename') or
                metadata.get('source') or
                metadata.get('document_id') or
                os.path.basename(metadata.get('file_path', '')) or
                f"Document_{i+1}"
            )

            # Clean file name
            if not file_name or file_name.strip() in ['Unknown File', '', 'Unknown']:
                file_path = metadata.get('file_path', '')
                if file_path:
                    file_name = os.path.basename(file_path)
                else:
                    # Generate a more meaningful name based on content
                    try:
                        content_preview = node.get_content()[:50].strip()
                        if content_preview:
                            content_hash = hashlib.md5(content_preview.encode()).hexdigest()[:6]
                            file_name = f"Source_{content_hash}"
                        else:
                            file_name = f"Document_{i+1}"
                    except Exception:
                        file_name = f"Document_{i+1}"

            # --- Page/Section/Info Extraction ---
            page_info = (
                metadata.get('page_label') or
                metadata.get('page') or
                metadata.get('page_number') or
                metadata.get('chunk_id', '')
            )

            section_info = metadata.get('section', metadata.get('chapter', ''))

            # Build the page/section string
            page_parts = []
            if page_info and str(page_info).strip() not in ['N/A', '', 'None']:
                page_str = str(page_info).strip()
                if page_str.isdigit():
                    page_parts.append(f"صفحة {page_info}" if detect_language(str(page_info)) == "arabic" else f"Page {page_info}")
                else:
                    page_parts.append(f"القسم {page_info}" if detect_language(str(page_info)) == "arabic" else f"Section {page_info}")
            
            if section_info and str(section_info).strip() not in ['N/A', '', 'None']:
                page_parts.append(f"الفصل {section_info}" if detect_language(str(section_info)) == "arabic" else f"Chapter {section_info}")
            
            # Only include score if it's meaningful (> 0.1) to avoid clutter
            if score > 0.1:
                page_parts.append(f"(الصلة: {score:.2f})")
            
            page_str = " | ".join(page_parts) if page_parts else "موقع غير محدد"

            # --- Lenient Filtering Logic ---
            # Only filter out nodes with extremely low scores or empty content
            content = node.get_content() if hasattr(node, 'get_content') else str(node)
            
            # Very basic filtering - only exclude truly empty or irrelevant content
            if len(content.strip()) < 10:  # Only filter very short content
                continue
                
            # Only apply score filter if we have meaningful scores
            if scores and max(scores) > 0.1 and score < MIN_RELEVANCE_FOR_CITATION:
                continue

            # Check if we've seen this file before
            if file_name in file_citations:
                # Keep the higher scoring chunk for this file
                if score > file_citations[file_name]['score']:
                    file_citations[file_name] = {
                        'page_str': page_str,
                        'score': score,
                        'content': content[:200] + "..." if len(content) > 200 else content
                    }
            else:
                # Add new file
                file_citations[file_name] = {
                    'page_str': page_str,
                    'score': score,
                    'content': content[:200] + "..." if len(content) > 200 else content
                }

        except Exception as e:
            logger.error(f"Error processing citation for node {i}: {e}")
            # Even with errors, try to add a basic citation
            try:
                fallback_name = f"مصدر_{i+1}"  # Arabic fallback
                if fallback_name not in file_citations:
                    file_citations[fallback_name] = {
                        'page_str': "خطأ في استخراج البيانات الوصفية",
                        'score': 0.0,
                        'content': "محتوى غير متاح"
                    }
            except Exception:
                continue

    # --- Ensure we always have citations if we had nodes ---
    if not file_citations and retrieved_nodes:
        # Fallback: create basic citations for all nodes
        logger.warning("No citations generated, creating fallback citations")
        for i, node_with_score in enumerate(retrieved_nodes[:3]):  # Limit to 3 fallbacks
            try:
                if isinstance(node_with_score, NodeWithScore):
                    node = node_with_score.node
                else:
                    node = node_with_score
                
                file_citations[f"مصدر_{i+1}"] = {
                    'page_str': "موقع غير محدد",
                    'score': 0.0,
                    'content': getattr(node, 'text', str(node))[:100] + "..."
                }
            except Exception:
                file_citations[f"مصدر_{i+1}"] = {
                    'page_str': "موقع غير محدد", 
                    'score': 0.0,
                    'content': "محتوى غير متاح"
                }

    # --- Generate Final Citation List ---
    if not file_citations:
        # Last resort fallback
        return [f"📄 مصدر 1 | موقع غير محدد"]
        
    # Sort by score descending, but always show at least one citation
    sorted_files = sorted(file_citations.items(), key=lambda item: item[1]['score'], reverse=True)

    citations = []
    for file_name, citation_data in sorted_files:
        # Format the citation string for this file
        citation = f"📄 {file_name}"
        if citation_data['page_str'] and citation_data['page_str'] != "موقع غير محدد":
            citation += f" | {citation_data['page_str']}"
        citations.append(citation)

    # Ensure we have at least one citation, limit to reasonable number
    if not citations:
        citations = [f"📄 مصدر 1 | موقع غير محدد"]
    elif len(citations) > 5:  # Limit to 5 sources max
        citations = citations[:5]

    return citations


def format_citations_safe_fallback(retrieved_nodes: List[NodeWithScore]) -> List[str]:
    """
    Ultra-safe fallback citation formatter that ALWAYS returns citations if nodes exist.
    Use this as a backup if the enhanced version fails.
    """
    if not retrieved_nodes:
        return []
    
    citations = []
    seen_files = set()
    
    for i, node_with_score in enumerate(retrieved_nodes):
        try:
            # Extract basic info safely
            if isinstance(node_with_score, NodeWithScore):
                node = node_with_score.node
                score = getattr(node_with_score, 'score', 0.0)
            else:
                node = node_with_score
                score = 0.0
            
            # Get file name with multiple fallbacks
            file_name = "مصدر غير محدد"  # Default Arabic
            if hasattr(node, 'metadata') and node.metadata:
                file_name = (
                    node.metadata.get('file_name') or
                    node.metadata.get('filename') or
                    f"مصدر_{i+1}"
                )
            
            # Avoid duplicates
            if file_name in seen_files:
                continue
            seen_files.add(file_name)
            
            # Create simple citation
            citation = f"📄 {file_name}"
            
            # Add page info if available
            if hasattr(node, 'metadata') and node.metadata:
                page_info = node.metadata.get('page_label') or node.metadata.get('page')
                if page_info and str(page_info).strip() not in ['N/A', '', 'None']:
                    citation += f" | صفحة {page_info}"
            
            citations.append(citation)
            
            # Limit to 3 sources for simplicity
            if len(citations) >= 3:
                break
                
        except Exception as e:
            logger.error(f"Error in fallback citation for node {i}: {e}")
            # Even if there's an error, add a basic citation
            citations.append(f"📄 مصدر_{i+1} | موقع غير محدد")
    
    # Ensure at least one citation
    if not citations:
        citations = [f"📄 مصدر 1 | موقع غير محدد"]
    
    return citations

def analyze_answer_completeness(retrieved_nodes: List[NodeWithScore], query: str) -> Dict[str, Any]:
    """
    Analyze if the answer is likely complete in the top-scoring source.
    This helps determine if multiple sources are actually needed.
    """
    if not retrieved_nodes:
        return {"is_complete": False, "primary_source": None, "confidence": 0.0}
    
    # Get the top node
    top_node = retrieved_nodes[0] if isinstance(retrieved_nodes[0], NodeWithScore) else None
    if not top_node:
        return {"is_complete": False, "primary_source": None, "confidence": 0.0}
    
    top_content = top_node.node.get_content()
    top_score = getattr(top_node, 'score', 0.0)
    
    # Analyze query complexity
    query_terms = set(re.findall(r'\w+', query.lower()))
    content_terms = set(re.findall(r'\w+', top_content.lower()))
    
    # Calculate coverage
    coverage = len(query_terms & content_terms) / max(len(query_terms), 1)
    
    # Check for list indicators in query
    list_indicators = ['list', 'all', 'every', 'each', 'جميع', 'كل', 'قائمة']
    is_list_query = any(indicator in query.lower() for indicator in list_indicators)
    
    # Determine completeness
    is_complete = (
        coverage > 0.6 and  # Good term coverage
        top_score > 0.5 and  # Decent confidence
        (not is_list_query or len(top_content) > 200)  # For list queries, need substantial content
    )
    
    return {
        "is_complete": is_complete,
        "primary_source": top_node.node.metadata.get('file_name', 'Unknown'),
        "confidence": float(top_score) if top_score is not None else 0.0,
        "coverage": coverage
    }

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
1. **الاعتماد الحصري على السياق:** استخدم فقط المعلومات الموجودة في الوثائق المقدمة أدناه. لا تسترجع إلا ما هو موجود، ولا تتخمين أبداً.
2. **الدقة في الاستخلاص:** عند طلب قوائم محددة، قدم *جميع* العناصر كما وردت تماماً في النص، مع الإشارة إلى رقم المادة والفقرة. اجمعها في ترتيب ظهورها أو حسب التصنيف إذا كان مناسباً.
3. **التحقق من الاكتمال:** ابحث في *كامل* السياق المقدم للتأكد من عدم تفويت أي معلومات ذات صلة. عد العناصر بدقة إذا طُلب عددها.
4. **المصطلحات :** استخدم المصطلحات كما وردت في النص دون تغيير أو استبدال.
5. **الشفافية:** إذا لم تجد معلومات كافية، أجب بوضوح: "لا توجد معلومات كافية في الوثائق المقدمة." لا تضف تفسيرات خارجية.
6. **البحث الكامل**: ابحث في *كل* السياق المقدم أدناه - اقرأ كل سطر بعناية.
7. **عدم التوقف المبكر**: لا تتوقف عند أول نتيجة - استمر حتى نهاية النص لجمع كل التفاصيل.
8. **القوائم الشاملة**: إذا طُلبت قائمة، يجب تضمين *جميع* العناصر - صفر استثناءات، وتجنب التكرار إلا إذا كان في النص.
9. **الترقيم الأصلي**: احتفظ بأرقام المواد والفقرات كما وردت تماماً.
10. **الاستشهاد الإجباري**: لكل نقطة، اذكر (المادة X، الفقرة Y) واقتبس العبارات الرئيسية مباشرة من النص.
11. **التعامل مع التكرار**: إذا ذكرت المعلومات في وثائق متعددة، استشهد بكل المصادر ذات الصلة.
**تنبيه هام:** فكر خطوة بخطوة: أولاً، اقرأ السياق كاملاً؛ ثانياً، حدد المعلومات ذات الصلة؛ ثالثاً، رتب الإجابة؛ أخيراً، راجع إجابتك للتأكد من الدقة والاكتمال وعدم نسيان أي عنصر موجود في السياق.

**السياق المقدم:**
{context_str}

**السؤال:**
{query_str}

**التعليمات الإضافية:**
- ابدأ مباشرة بالإجابة دون مقدمات أو تفكير مرئي.
- استخدم نقاط مرقمة لكل عنصر أو نقطة مع الإشارة إلى مصدرها.
- ضمن الاكتمال والدقة باستخدام اقتباسات مباشرة حيث يناسب.
- لا تضف تفسيرات أو تعليقات إضافية غير موجودة في النص.

**الإجابة:**"""

    else:  # English
        return f"""You are an intelligent assistant specialized in answering questions based *exclusively* on the provided information. Follow these rules with strict precision:

**Core Rules:**
1. **Exclusive Context Reliance:** Use only information explicitly present in the documents provided below. Retrieve only what is there, no guessing ever.
2. **Extraction Precision:** When specific lists are requested, provide *all* elements exactly as stated in the text, with proper article and paragraph references. Present them in order of appearance or grouped by category if applicable.
3. **Completeness Verification:** Search the *entire* provided context to ensure no relevant information is missed. Count items precisely if a count is requested.
4. **Terminology:** Use terms exactly as they appear in the text without alteration or substitution.
5. **Transparency:** If insufficient information is found, clearly respond: "Insufficient information in provided documents." Do not add external interpretations.
6. **Complete Search**: Search through *ALL* the context provided below - read every line carefully.
7. **No Early Stopping**: Don't stop at the first result - continue to the end of text to gather all details.
8. **Exhaustive Lists**: If a list is requested, include *ALL* elements - zero exceptions, and avoid duplication unless in the text.
9. **Original Numbering**: Preserve article and paragraph numbers exactly as stated.
10. **Mandatory Citations**: For each point, include (Article X, Paragraph Y) and quote key phrases directly from the text.
11. **Handling Duplicates**: If information is mentioned in multiple documents, cite all relevant sources.
**Important:** Think step-by-step: First, read the full context; Second, identify relevant information; Third, organize the answer; Finally, review your answer to ensure accuracy, completeness, and that no element from context is forgotten.

**Provided Context:**
{context_str}

**Question:**
{query_str}

**Additional Instructions:**
- Start directly with the answer, no preambles or visible thinking.
- Use numbered points for each item or point with source references.
- Ensure completeness and accuracy using direct quotes where appropriate.
- Do not add interpretations or comments not found in the text.

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

# Additional utility functions for  RAG

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