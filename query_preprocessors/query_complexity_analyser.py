# query_complexity_analyser.py

from query_preprocessors.base_preprocessor import BasePreprocessor
import re


class QueryComplexityAnalyser(BasePreprocessor):

    order = 90

    def __init__(self):
        super().__init__()

    def process(self, context):
        q = context.query.lower()
        tokens = q.split()
        length = len(tokens)

        complexity = "medium"

        if length < 5:
            complexity = "low"
        elif length > 18:
            complexity = "high"

        if " and " in q or " or " in q or "," in q:
            complexity = "multi-entity"

        if re.search(r"\d+", q):
            complexity = "structured"

        if context.buildings and len(context.buildings) > 1:
            complexity = "multi-building"

        context.complexity = complexity
        self.logger.info(f"Query complexity: {complexity}")
