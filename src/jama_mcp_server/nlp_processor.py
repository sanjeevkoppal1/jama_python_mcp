"""
NLP Processing Pipeline for Jama Requirements

Comprehensive natural language processing for:
- Business rule extraction (mortgage rules, conditions, etc.)
- Requirement classification (functional/non-functional)
- Entity extraction and relationship mapping
- Semantic enrichment for search and analysis
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

import spacy
import pandas as pd
import numpy as np
from spacy.tokens import Doc, Span
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob

logger = logging.getLogger(__name__)


class RequirementType(Enum):
    """Classification of requirement types."""
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    BUSINESS_RULE = "business_rule"
    CONSTRAINT = "constraint"
    INTERFACE = "interface"
    QUALITY = "quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USABILITY = "usability"
    UNKNOWN = "unknown"


class BusinessRuleType(Enum):
    """Types of business rules that can be extracted."""
    CONDITIONAL = "conditional"  # If-then rules
    CALCULATION = "calculation"  # Mathematical formulas
    CONSTRAINT = "constraint"    # Restrictions and limits
    VALIDATION = "validation"    # Data validation rules
    WORKFLOW = "workflow"        # Process rules
    POLICY = "policy"           # Organizational policies
    REGULATION = "regulation"   # Compliance rules


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from requirements."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    context: str = ""


@dataclass
class BusinessRule:
    """Represents an extracted business rule."""
    text: str
    rule_type: BusinessRuleType
    condition: Optional[str] = None
    action: Optional[str] = None
    entities: List[ExtractedEntity] = field(default_factory=list)
    confidence: float = 0.0
    source_requirement_id: Optional[str] = None


@dataclass
class ProcessedRequirement:
    """Enhanced requirement with NLP analysis results."""
    original_id: str
    text: str
    classification: RequirementType
    business_rules: List[BusinessRule] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sentiment: float = 0.0
    complexity_score: float = 0.0
    embedding: Optional[np.ndarray] = None
    similar_requirements: List[str] = field(default_factory=list)


class NLPProcessor:
    """
    Comprehensive NLP processor for Jama requirements analysis.
    
    Features:
    - Multi-model processing (spaCy, transformers, sentence-transformers)
    - Business rule extraction with pattern matching
    - Requirement classification
    - Entity recognition and relationship mapping
    - Semantic embedding generation
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        sentence_model: str = "all-MiniLM-L6-v2",
        enable_gpu: bool = False,
        batch_size: int = 32
    ):
        self.spacy_model_name = spacy_model
        self.sentence_model_name = sentence_model
        self.enable_gpu = enable_gpu
        self.batch_size = batch_size
        
        # Model placeholders
        self.nlp = None
        self.sentence_model = None
        self.classifier = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Business rule patterns
        self.business_rule_patterns = self._load_business_rule_patterns()
        
        # Domain-specific entities for requirements
        self.domain_entities = {
            "MORTGAGE": ["mortgage", "loan", "interest rate", "principal", "down payment", 
                        "credit score", "debt-to-income", "appraisal", "closing costs"],
            "CONDITION": ["if", "when", "unless", "provided that", "subject to", 
                         "in case of", "depending on", "conditional upon"],
            "CONSTRAINT": ["must", "shall", "required", "mandatory", "prohibited", 
                          "not allowed", "restricted", "limited to", "maximum", "minimum"],
            "CALCULATION": ["calculate", "formula", "percentage", "rate", "amount", 
                           "total", "sum", "average", "multiply", "divide"],
            "TEMPORAL": ["daily", "monthly", "annually", "quarterly", "within", 
                        "before", "after", "during", "deadline", "expiry"]
        }

    async def initialize(self) -> None:
        """Initialize all NLP models and components."""
        logger.info("Initializing NLP processor...")
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(self.spacy_model_name)
            logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
        except OSError:
            logger.error(f"spaCy model {self.spacy_model_name} not found. Please install it.")
            raise
        
        # Add custom business rule patterns to spaCy
        self._add_custom_patterns()
        
        # Load sentence transformer model
        try:
            device = "cuda" if self.enable_gpu else "cpu"
            self.sentence_model = SentenceTransformer(
                self.sentence_model_name, 
                device=device
            )
            logger.info(f"Loaded sentence transformer: {self.sentence_model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise
        
        # Initialize classification pipeline
        try:
            self.classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",  # Placeholder - would use custom model
                device=0 if self.enable_gpu else -1
            )
        except Exception as e:
            logger.warning(f"Could not load classification model: {e}")
            self.classifier = None
        
        logger.info("NLP processor initialized successfully")

    def _load_business_rule_patterns(self) -> Dict[BusinessRuleType, List[Dict]]:
        """Load business rule extraction patterns."""
        return {
            BusinessRuleType.CONDITIONAL: [
                {"pattern": r"(?i)if\s+(.+?)\s+then\s+(.+)", "groups": ["condition", "action"]},
                {"pattern": r"(?i)when\s+(.+?),?\s+(?:then\s+)?(.+)", "groups": ["condition", "action"]},
                {"pattern": r"(?i)provided\s+that\s+(.+?),?\s+(.+)", "groups": ["condition", "action"]},
                {"pattern": r"(?i)in\s+case\s+(?:of\s+)?(.+?),?\s+(.+)", "groups": ["condition", "action"]},
            ],
            BusinessRuleType.CONSTRAINT: [
                {"pattern": r"(?i)must\s+not\s+(.+)", "groups": ["constraint"]},
                {"pattern": r"(?i)(?:shall|must|required to)\s+(.+)", "groups": ["constraint"]},
                {"pattern": r"(?i)(?:minimum|maximum|at least|no more than)\s+(.+)", "groups": ["constraint"]},
                {"pattern": r"(?i)(?:prohibited|not allowed|forbidden)\s+(.+)", "groups": ["constraint"]},
            ],
            BusinessRuleType.CALCULATION: [
                {"pattern": r"(?i)(?:calculate|compute|determine)\s+(.+?)\s+(?:as|by|using)\s+(.+)", "groups": ["target", "formula"]},
                {"pattern": r"(?i)(.+?)\s+(?:is calculated as|equals|=)\s+(.+)", "groups": ["target", "formula"]},
                {"pattern": r"(?i)(?:interest rate|rate|percentage)\s+(?:of|is)\s+(.+)", "groups": ["formula"]},
            ],
            BusinessRuleType.VALIDATION: [
                {"pattern": r"(?i)(?:validate|verify|check)\s+(?:that\s+)?(.+)", "groups": ["validation"]},
                {"pattern": r"(?i)(.+?)\s+(?:must be|should be)\s+(?:valid|verified|checked)", "groups": ["validation"]},
            ],
            BusinessRuleType.POLICY: [
                {"pattern": r"(?i)(?:policy|rule|regulation)\s+(?:states|requires|mandates)\s+(?:that\s+)?(.+)", "groups": ["policy"]},
                {"pattern": r"(?i)according\s+to\s+(?:policy|regulation|rule)\s+(.+)", "groups": ["policy"]},
            ]
        }

    def _add_custom_patterns(self) -> None:
        """Add custom entity recognition patterns to spaCy."""
        # Create custom entity ruler
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.get_pipe("entity_ruler")
        
        patterns = []
        
        # Add domain-specific entity patterns
        for entity_type, terms in self.domain_entities.items():
            for term in terms:
                patterns.append({"label": entity_type, "pattern": term.lower()})
                patterns.append({"label": entity_type, "pattern": term.title()})
        
        # Add business rule indicators
        rule_indicators = [
            {"label": "RULE_INDICATOR", "pattern": [{"LOWER": {"IN": ["if", "when", "unless", "provided"]}}]},
            {"label": "CONSTRAINT_INDICATOR", "pattern": [{"LOWER": {"IN": ["must", "shall", "required", "prohibited"]}}]},
            {"label": "CALCULATION_INDICATOR", "pattern": [{"LOWER": {"IN": ["calculate", "compute", "formula", "rate"]}}]},
        ]
        
        patterns.extend(rule_indicators)
        ruler.add_patterns(patterns)

    async def process_requirement(self, text: str, requirement_id: str) -> ProcessedRequirement:
        """
        Process a single requirement with comprehensive NLP analysis.
        
        Args:
            text: Requirement text to process
            requirement_id: Unique identifier for the requirement
            
        Returns:
            ProcessedRequirement with all analysis results
        """
        # Run NLP processing in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Process with spaCy
        doc = await loop.run_in_executor(self.executor, self._process_with_spacy, text)
        
        # Extract entities
        entities = self._extract_entities(doc)
        
        # Extract business rules
        business_rules = await self._extract_business_rules(text, requirement_id)
        
        # Classify requirement type
        classification = await self._classify_requirement(text, doc)
        
        # Generate keywords
        keywords = self._extract_keywords(doc)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(text)
        
        # Calculate complexity
        complexity = self._calculate_complexity(doc)
        
        # Generate embedding
        embedding = await loop.run_in_executor(
            self.executor, 
            self._generate_embedding, 
            text
        )
        
        return ProcessedRequirement(
            original_id=requirement_id,
            text=text,
            classification=classification,
            business_rules=business_rules,
            entities=entities,
            keywords=keywords,
            sentiment=sentiment,
            complexity_score=complexity,
            embedding=embedding,
            similar_requirements=[]  # Will be populated during similarity analysis
        )

    def _process_with_spacy(self, text: str) -> Doc:
        """Process text with spaCy model."""
        return self.nlp(text)

    def _extract_entities(self, doc: Doc) -> List[ExtractedEntity]:
        """Extract entities from spaCy doc."""
        entities = []
        
        for ent in doc.ents:
            entity = ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,  # spaCy doesn't provide confidence scores by default
                context=self._get_entity_context(doc, ent)
            )
            entities.append(entity)
        
        return entities

    def _get_entity_context(self, doc: Doc, ent: Span) -> str:
        """Get surrounding context for an entity."""
        # Get 5 tokens before and after the entity
        start = max(0, ent.start - 5)
        end = min(len(doc), ent.end + 5)
        context_span = doc[start:end]
        return context_span.text

    async def _extract_business_rules(self, text: str, requirement_id: str) -> List[BusinessRule]:
        """Extract business rules using pattern matching and NLP."""
        rules = []
        
        for rule_type, patterns in self.business_rule_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                groups = pattern_info["groups"]
                
                matches = re.finditer(pattern, text)
                for match in matches:
                    rule_text = match.group(0)
                    
                    # Extract condition and action based on groups
                    condition = None
                    action = None
                    
                    if "condition" in groups and len(match.groups()) >= 1:
                        condition = match.group(1).strip()
                    if "action" in groups and len(match.groups()) >= 2:
                        action = match.group(2).strip()
                    elif "constraint" in groups and len(match.groups()) >= 1:
                        action = match.group(1).strip()
                    elif "formula" in groups:
                        if len(match.groups()) >= 2:
                            action = match.group(2).strip()
                        else:
                            action = match.group(1).strip()
                    
                    # Calculate confidence based on pattern strength
                    confidence = self._calculate_rule_confidence(rule_text, rule_type)
                    
                    rule = BusinessRule(
                        text=rule_text,
                        rule_type=rule_type,
                        condition=condition,
                        action=action,
                        confidence=confidence,
                        source_requirement_id=requirement_id
                    )
                    
                    rules.append(rule)
        
        return rules

    def _calculate_rule_confidence(self, rule_text: str, rule_type: BusinessRuleType) -> float:
        """Calculate confidence score for extracted business rule."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on rule indicators
        indicators = {
            BusinessRuleType.CONDITIONAL: ["if", "when", "then", "provided", "unless"],
            BusinessRuleType.CONSTRAINT: ["must", "shall", "required", "prohibited", "not allowed"],
            BusinessRuleType.CALCULATION: ["calculate", "formula", "rate", "percentage", "equals"],
            BusinessRuleType.VALIDATION: ["validate", "verify", "check", "ensure"],
            BusinessRuleType.POLICY: ["policy", "rule", "regulation", "according to"]
        }
        
        rule_indicators = indicators.get(rule_type, [])
        lower_text = rule_text.lower()
        
        for indicator in rule_indicators:
            if indicator in lower_text:
                confidence += 0.1
        
        # Boost for specific domain terms (mortgage example)
        mortgage_terms = ["mortgage", "loan", "interest", "credit", "down payment", "appraisal"]
        for term in mortgage_terms:
            if term in lower_text:
                confidence += 0.05
        
        return min(confidence, 1.0)

    async def _classify_requirement(self, text: str, doc: Doc) -> RequirementType:
        """Classify requirement type using rules and ML."""
        # Rule-based classification first
        lower_text = text.lower()
        
        # Check for business rule indicators
        rule_indicators = ["if", "when", "must", "shall", "calculate", "formula"]
        if any(indicator in lower_text for indicator in rule_indicators):
            return RequirementType.BUSINESS_RULE
        
        # Check for non-functional indicators
        nf_indicators = ["performance", "security", "usability", "reliability", "scalability"]
        if any(indicator in lower_text for indicator in nf_indicators):
            return RequirementType.NON_FUNCTIONAL
        
        # Check for constraint indicators
        constraint_indicators = ["constraint", "limitation", "restriction", "prohibited"]
        if any(indicator in lower_text for indicator in constraint_indicators):
            return RequirementType.CONSTRAINT
        
        # Check for interface indicators
        interface_indicators = ["interface", "api", "integration", "connection"]
        if any(indicator in lower_text for indicator in interface_indicators):
            return RequirementType.INTERFACE
        
        # Default to functional if no specific indicators
        return RequirementType.FUNCTIONAL

    def _extract_keywords(self, doc: Doc) -> List[str]:
        """Extract important keywords from the text."""
        keywords = []
        
        # Extract noun phrases and important entities
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 2 and chunk.root.pos_ in ["NOUN", "PROPN"]:
                keywords.append(chunk.text.lower())
        
        # Extract important single tokens
        for token in doc:
            if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:20]  # Limit to top 20 keywords

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment polarity of the requirement text."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0

    def _calculate_complexity(self, doc: Doc) -> float:
        """Calculate complexity score based on linguistic features."""
        complexity = 0.0
        
        # Sentence complexity
        sentences = list(doc.sents)
        avg_sentence_length = np.mean([len(sent) for sent in sentences]) if sentences else 0
        complexity += min(avg_sentence_length / 20, 1.0) * 0.3
        
        # Vocabulary complexity (unique tokens)
        unique_tokens = len(set(token.lemma_.lower() for token in doc if not token.is_punct))
        complexity += min(unique_tokens / 50, 1.0) * 0.3
        
        # Syntactic complexity (dependency depth)
        max_depth = 0
        for token in doc:
            depth = self._get_dependency_depth(token)
            max_depth = max(max_depth, depth)
        complexity += min(max_depth / 10, 1.0) * 0.2
        
        # Business rule complexity
        rule_indicators = ["if", "when", "unless", "provided", "calculate", "formula"]
        rule_count = sum(1 for indicator in rule_indicators if indicator in doc.text.lower())
        complexity += min(rule_count / 5, 1.0) * 0.2
        
        return complexity

    def _get_dependency_depth(self, token) -> int:
        """Calculate the dependency tree depth for a token."""
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
            if depth > 20:  # Prevent infinite loops
                break
        return depth

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate semantic embedding for the text."""
        try:
            embedding = self.sentence_model.encode([text])[0]
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2

    async def process_requirements_batch(
        self, 
        requirements: List[Tuple[str, str]]  # (text, id) pairs
    ) -> List[ProcessedRequirement]:
        """
        Process multiple requirements in batch for efficiency.
        
        Args:
            requirements: List of (text, requirement_id) tuples
            
        Returns:
            List of ProcessedRequirement objects
        """
        logger.info(f"Processing batch of {len(requirements)} requirements")
        
        tasks = []
        for text, req_id in requirements:
            task = self.process_requirement(text, req_id)
            tasks.append(task)
        
        # Process in smaller batches to manage memory
        batch_size = self.batch_size
        results = []
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
        
        logger.info(f"Completed processing {len(results)} requirements")
        return results

    async def find_similar_requirements(
        self, 
        processed_requirements: List[ProcessedRequirement],
        similarity_threshold: float = 0.7
    ) -> None:
        """
        Find similar requirements using semantic embeddings.
        
        Args:
            processed_requirements: List of processed requirements
            similarity_threshold: Minimum similarity score
        """
        logger.info("Computing requirement similarities...")
        
        # Extract embeddings
        embeddings = []
        req_ids = []
        
        for req in processed_requirements:
            if req.embedding is not None:
                embeddings.append(req.embedding)
                req_ids.append(req.original_id)
        
        if len(embeddings) < 2:
            logger.warning("Not enough requirements with embeddings for similarity analysis")
            return
        
        # Compute similarity matrix
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Find similar requirements for each requirement
        req_dict = {req.original_id: req for req in processed_requirements}
        
        for i, req_id in enumerate(req_ids):
            similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
            similar_req_ids = [req_ids[j] for j in similar_indices if j != i]
            req_dict[req_id].similar_requirements = similar_req_ids
        
        logger.info(f"Completed similarity analysis for {len(req_ids)} requirements")

    async def search_business_rules(
        self,
        processed_requirements: List[ProcessedRequirement],
        query: str,
        rule_types: Optional[List[BusinessRuleType]] = None,
        min_confidence: float = 0.5
    ) -> List[BusinessRule]:
        """
        Search for business rules matching a query.
        
        Args:
            processed_requirements: List of processed requirements
            query: Search query (e.g., "mortgage rules", "interdiction conditions")
            rule_types: Filter by specific rule types
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching business rules
        """
        logger.info(f"Searching business rules for query: {query}")
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query.lower())
        
        matching_rules = []
        
        for req in processed_requirements:
            for rule in req.business_rules:
                # Filter by rule type if specified
                if rule_types and rule.rule_type not in rule_types:
                    continue
                
                # Filter by confidence
                if rule.confidence < min_confidence:
                    continue
                
                # Calculate semantic similarity with query
                rule_embedding = self._generate_embedding(rule.text.lower())
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    rule_embedding.reshape(1, -1)
                )[0][0]
                
                # Also check for keyword matches
                query_tokens = set(query.lower().split())
                rule_tokens = set(rule.text.lower().split())
                keyword_overlap = len(query_tokens.intersection(rule_tokens)) / len(query_tokens)
                
                # Combined score
                combined_score = (similarity * 0.7) + (keyword_overlap * 0.3)
                
                if combined_score > 0.3:  # Threshold for relevance
                    # Add the combined score to the rule for ranking
                    rule_copy = BusinessRule(
                        text=rule.text,
                        rule_type=rule.rule_type,
                        condition=rule.condition,
                        action=rule.action,
                        entities=rule.entities,
                        confidence=rule.confidence,
                        source_requirement_id=rule.source_requirement_id
                    )
                    # Store the search relevance score
                    rule_copy.search_score = combined_score
                    matching_rules.append(rule_copy)
        
        # Sort by relevance score
        matching_rules.sort(key=lambda r: getattr(r, 'search_score', 0), reverse=True)
        
        logger.info(f"Found {len(matching_rules)} matching business rules")
        return matching_rules

    async def close(self) -> None:
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("NLP processor closed")


# Utility functions for easy access

async def create_nlp_processor(
    spacy_model: str = "en_core_web_sm",
    sentence_model: str = "all-MiniLM-L6-v2",
    enable_gpu: bool = False
) -> NLPProcessor:
    """
    Create and initialize NLP processor.
    
    Args:
        spacy_model: spaCy model name
        sentence_model: Sentence transformer model name  
        enable_gpu: Whether to use GPU acceleration
        
    Returns:
        Initialized NLPProcessor
    """
    processor = NLPProcessor(
        spacy_model=spacy_model,
        sentence_model=sentence_model,
        enable_gpu=enable_gpu
    )
    
    await processor.initialize()
    return processor