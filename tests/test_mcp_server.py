from types import SimpleNamespace

from insurance_rag_assistant.mcp_server import server


class FakeSearch:
    def search(self, query):
        chunk = SimpleNamespace(
            rank=1, chunk_id="doc:sec:001", document_id="doc",
            document_name="Doc", section_title="Sec", text="Texte",
            score=0.8, rerank_score=3.2,
        )
        return SimpleNamespace(retrieval_sufficient=True, results=[chunk])


def test_search_documents_shape(monkeypatch):
    fake = SimpleNamespace(search_service=FakeSearch())
    monkeypatch.setattr(server, "get_evidence_retrieval_service", lambda: fake)
    out = server.search_documents("patch management", top_k=3)
    assert out["retrieval_sufficient"] is True
    assert out["results"][0]["chunk_id"] == "doc:sec:001"
