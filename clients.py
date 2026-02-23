#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client initialisation for Pinecone and OpenAI.
"""

import os
from typing import Optional
from pinecone import Pinecone
from openai import OpenAI


_pc: Optional[Pinecone] = None
_oai: Optional[OpenAI] = None


def get_pc() -> Pinecone:
    """
    Lazy-load Pinecone client.
    Only creates client when first needed.
    """
    global _pc
    if _pc is not None:
        return _pc

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. "
            "Set it in the environment or mock get_pc() during tests."
        )

    _pc = Pinecone(api_key=api_key)
    return _pc


def get_oai() -> OpenAI:
    global _oai
    if _oai is not None:
        return _oai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    _oai = OpenAI(api_key=api_key)
    return _oai
