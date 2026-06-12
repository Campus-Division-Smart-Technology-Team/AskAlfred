from types import SimpleNamespace

import ingest.utils as ingest_utils
from auth.access_control import filter_authorized_structured_matches
from query_core.query_context import DENY_ALL_TENANT_ID, build_access_filter
from search_core import structured_queries
from search_core.search_instructions import SearchInstructions
from search_core.search_router import execute
from search_core.search_utils import search_one_index


def test_build_access_filter_is_empty_for_anonymous_sessions():
    access_filter = build_access_filter(
        tenant_id=None,
        user_roles=(),
        authenticated=False,
    )

    assert access_filter == {}


def test_build_access_filter_scopes_authenticated_user_to_tenant():
    access_filter = build_access_filter(
        tenant_id="tenant-123",
        user_roles=(),
        authenticated=True,
    )

    assert access_filter == {"tenant_id": {"$eq": "tenant-123"}}


def test_build_access_filter_adds_role_constraint_when_roles_present():
    access_filter = build_access_filter(
        tenant_id="tenant-123",
        user_roles=("viewer", ""),
        authenticated=True,
    )

    assert access_filter == {
        "$and": [
            {"tenant_id": {"$eq": "tenant-123"}},
            {"allowed_roles": {"$in": ["viewer"]}},
        ]
    }


def test_build_access_filter_denies_authenticated_user_without_tenant():
    access_filter = build_access_filter(
        tenant_id=None,
        user_roles=("viewer",),
        authenticated=True,
    )

    assert access_filter == {"tenant_id": {"$eq": DENY_ALL_TENANT_ID}}


def test_search_router_passes_access_filter_to_semantic_search(monkeypatch):
    captured = {}

    def fake_semantic_search(query, top_k, building_filter=None, access_filter=None):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["building_filter"] = building_filter
        captured["access_filter"] = access_filter
        return [], "", "", False

    monkeypatch.setattr("search_core.search_router.semantic_search", fake_semantic_search)

    execute(
        SearchInstructions(
            type="semantic",
            query="Show me the FRA",
            top_k=5,
            building="Senate House",
            access_filter={"tenant_id": {"$eq": "tenant-123"}},
        )
    )

    assert captured == {
        "query": "Show me the FRA",
        "top_k": 5,
        "building_filter": "Senate House",
        "access_filter": {"tenant_id": {"$eq": "tenant-123"}},
    }


def test_search_one_index_combines_building_and_access_filters(monkeypatch):
    captured = {}

    monkeypatch.setattr("search_core.search_utils.open_index", lambda idx_name: object())
    monkeypatch.setattr(
        "search_core.search_utils.get_index_config",
        lambda idx_name: {"model": "text-embedding-3-small"},
    )
    monkeypatch.setattr(
        "search_core.search_utils._namespaces_to_search",
        lambda idx, idx_name: [None],
    )
    monkeypatch.setattr(
        "search_core.search_utils.create_building_metadata_filter",
        lambda building_name: {
            "canonical_building_name": {"$eq": building_name},
        },
    )
    monkeypatch.setattr(
        "search_core.search_utils.normalise_matches",
        lambda raw: [],
    )
    monkeypatch.setattr(
        "search_core.search_utils.embed_texts",
        lambda texts, model: [[0.0]],
    )

    def fake_vector_query(
        idx, namespace, query, k, embed_model, metadata_filter=None, query_vector=None
    ):
        captured["metadata_filter"] = metadata_filter
        return {"matches": []}

    monkeypatch.setattr("search_core.search_utils.vector_query", fake_vector_query)

    search_one_index(
        "test-index",
        "Which FRA applies?",
        building_filter="Senate House",
        access_filter={"tenant_id": {"$eq": "tenant-123"}},
    )

    assert captured["metadata_filter"] == {
        "$and": [
            {"tenant_id": {"$eq": "tenant-123"}},
            {"canonical_building_name": {"$eq": "Senate House"}},
        ]
    }


def test_search_one_index_filters_semantic_hits_missing_acl(monkeypatch):
    monkeypatch.setattr("search_core.search_utils.open_index", lambda idx_name: object())
    monkeypatch.setattr(
        "search_core.search_utils.get_index_config",
        lambda idx_name: {"model": "text-embedding-3-small"},
    )
    monkeypatch.setattr(
        "search_core.search_utils._namespaces_to_search",
        lambda idx, idx_name: [None],
    )
    monkeypatch.setattr(
        "search_core.search_utils.embed_texts",
        lambda texts, model: [[0.0]],
    )
    monkeypatch.setattr(
        "search_core.search_utils.normalise_matches",
        lambda raw: [
            {
                "metadata": {
                    "tenant_id": "tenant-123",
                    "access_level": "pilot_internal",
                    "allowed_roles": ["viewer"],
                    "key": "allowed",
                }
            },
            {
                "metadata": {
                    "tenant_id": "tenant-123",
                    "key": "missing-acl",
                }
            },
        ],
    )
    monkeypatch.setattr(
        "search_core.search_utils.vector_query",
        lambda idx, namespace, query, k, embed_model, metadata_filter=None, query_vector=None: {
            "matches": []
        },
    )

    hits = search_one_index(
        "test-index",
        "Which FRA applies?",
        access_filter={"tenant_id": {"$eq": "tenant-123"}},
    )

    assert hits == [
        {
            "metadata": {
                "tenant_id": "tenant-123",
                "access_level": "pilot_internal",
                "allowed_roles": ["viewer"],
                "key": "allowed",
            },
            "index": "test-index",
            "namespace": None,
        }
    ]


def test_filter_authorized_structured_matches_fails_closed_on_missing_acl():
    matches = [
        {
            "metadata": {
                "tenant_id": "tenant-123",
                "access_level": "pilot_internal",
                "allowed_roles": ["viewer"],
            }
        },
        {
            "metadata": {
                "tenant_id": "tenant-123",
                "access_level": "pilot_internal",
            }
        },
        {
            "metadata": {
                "tenant_id": "other-tenant",
                "access_level": "pilot_internal",
                "allowed_roles": ["viewer"],
            }
        },
    ]

    authorised = filter_authorized_structured_matches(
        matches,
        access_filter={"tenant_id": {"$eq": "tenant-123"}},
    )

    assert authorised == [
        {
            "metadata": {
                "tenant_id": "tenant-123",
                "access_level": "pilot_internal",
                "allowed_roles": ["viewer"],
            }
        }
    ]


def test_filter_authorized_structured_matches_keeps_legacy_vectors_when_unscoped():
    """Unscoped (anonymous/dev) sessions still see pre-ACL legacy vectors."""
    matches = [
        {"metadata": {"key": "legacy-no-acl"}},
    ]

    assert filter_authorized_structured_matches(matches, access_filter=None) == matches
    assert filter_authorized_structured_matches(matches, access_filter={}) == matches


def test_query_index_with_batches_combines_filters_and_filters_acl(monkeypatch):
    captured = {}

    class FakeIndex:
        def describe_index_stats(self):
            return {
                "namespaces": {None: {"vector_count": 2}},
                "dimension": 3,
            }

    monkeypatch.setattr(structured_queries, "open_index", lambda idx_name: FakeIndex())
    def fake_query_all_chunks(
        index, namespace, query_vector, filter_dict, top_k, include_metadata
    ):
        captured["filter_dict"] = filter_dict
        return [
            {
                "metadata": {
                    "key": "allowed",
                    "tenant_id": "tenant-123",
                    "access_level": "pilot_internal",
                    "allowed_roles": ["viewer"],
                    "canonical_building_name": "Senate House",
                }
            },
            {
                "metadata": {
                    "key": "missing-acl",
                    "tenant_id": "tenant-123",
                    "canonical_building_name": "Senate House",
                }
            },
        ]

    monkeypatch.setattr(
        structured_queries,
        "query_all_chunks",
        fake_query_all_chunks,
    )

    matches = structured_queries._query_index_with_batches(
        idx_name="test-index",
        namespace=None,
        filter_dict={"canonical_building_name": {"$eq": "Senate House"}},
        access_filter={"tenant_id": {"$eq": "tenant-123"}},
        top_k=10,
    )

    assert captured["filter_dict"] == {
        "$and": [
            {"canonical_building_name": {"$eq": "Senate House"}},
            {"tenant_id": {"$eq": "tenant-123"}},
        ]
    }
    assert matches == [
        {
            "metadata": {
                "key": "allowed",
                "tenant_id": "tenant-123",
                "access_level": "pilot_internal",
                "allowed_roles": ["viewer"],
                "canonical_building_name": "Senate House",
            }
        }
    ]


def test_validate_with_truncation_applies_ingest_acl_defaults(monkeypatch):
    monkeypatch.setattr(ingest_utils, "INGEST_DEFAULT_TENANT_ID", "tenant-123")
    monkeypatch.setattr(ingest_utils, "INGEST_DEFAULT_ACCESS_LEVEL", "pilot_internal")
    monkeypatch.setattr(ingest_utils, "INGEST_DEFAULT_ALLOWED_ROLES", ("pilot_user",))

    ctx = SimpleNamespace(
        config=SimpleNamespace(
            max_metadata_size=10000,
            max_metadata_text_tokens=1000,
        ),
        encoder=SimpleNamespace(
            encode=lambda text: list(text),
            decode=lambda tokens: "".join(tokens),
        ),
    )
    metadata = {
        "source_path": "docs",
        "key": "doc-1",
        "source": "doc.txt",
        "text": "hello",
        "document_type": "operational_doc",
    }

    valid, reason = ingest_utils.validate_with_truncation(ctx, metadata)

    assert valid is True
    assert reason is None
    assert metadata["tenant_id"] == "tenant-123"
    assert metadata["access_level"] == "pilot_internal"
    assert metadata["allowed_roles"] == ["pilot_user"]


def test_validate_with_truncation_fails_without_acl_defaults(monkeypatch):
    monkeypatch.setattr(ingest_utils, "INGEST_DEFAULT_TENANT_ID", None)
    monkeypatch.setattr(ingest_utils, "INGEST_DEFAULT_ACCESS_LEVEL", "")
    monkeypatch.setattr(ingest_utils, "INGEST_DEFAULT_ALLOWED_ROLES", ())

    ctx = SimpleNamespace(
        config=SimpleNamespace(
            max_metadata_size=10000,
            max_metadata_text_tokens=1000,
        ),
        encoder=SimpleNamespace(
            encode=lambda text: list(text),
            decode=lambda tokens: "".join(tokens),
        ),
    )
    metadata = {
        "source_path": "docs",
        "key": "doc-1",
        "source": "doc.txt",
        "text": "hello",
        "document_type": "operational_doc",
    }

    valid, reason = ingest_utils.validate_with_truncation(ctx, metadata)

    assert valid is False
    assert reason == "Missing required ACL metadata fields"
