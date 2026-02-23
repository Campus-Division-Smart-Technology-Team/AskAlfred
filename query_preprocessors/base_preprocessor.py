# base_preprocessor.py

import logging


class BasePreprocessor:
    """Base class for query preprocessors."""

    order = 100

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def name(self):
        return self.__class__.__name__

    def should_run(self, context) -> bool:
        return True

    def process(self, context) -> None:
        raise NotImplementedError


class CachingPreprocessor(BasePreprocessor):
    """Adds optional cache skipping rules."""

    cache_key = None

    def should_run(self, context) -> bool:
        if self.cache_key and context.get_from_cache(self.cache_key):
            return False
        return True

    def mark_done(self, context):
        if self.cache_key:
            context.add_to_cache(self.cache_key, True)

    def process(self, context) -> None:
        raise NotImplementedError
