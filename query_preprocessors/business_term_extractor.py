# business_term_extractor.py

from query_preprocessors.base_preprocessor import CachingPreprocessor
from business_terms import BusinessTermMapper


class BusinessTermExtractor(CachingPreprocessor):
    """
    Detects ALL business terms in the query.
    Adds:
        context.business_terms = list of all terms
        context.document_types = all doc types
        context.document_type  = first doc type (legacy)
    """

    order = 30
    cache_key = "business_terms_extracted"

    def __init__(self):
        super().__init__()
        self.mapper = BusinessTermMapper()

    def process(self, context):
        terms = self.mapper.detect_business_terms(context.query)
        if not terms:
            return

        context.business_terms = terms

        doc_terms = [t for t in terms if t.get("document_type")]
        if doc_terms:
            context.document_types = doc_terms
            context.document_type = doc_terms[0]["document_type"]

        self.mark_done(context)

        self.logger.info(
            f"Detected business terms: {', '.join(t['term'] for t in terms)}")
