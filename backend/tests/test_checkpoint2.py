import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.collectors import hacker_news as hn
from app.main import app


def test_echo_and_existing_routes():
    with TestClient(app) as client:
        payload = {"company": "Microsoft", "role": "Senior Software Engineer",
                   "experience": 10, "location": "India"}
        response = client.post("/api/analyze", json=payload)
        assert response.status_code == 200
        assert response.json() == payload
        del payload["location"]
        assert client.post("/api/analyze", json=payload).json() == {**payload, "location": None}
        assert client.get("/").json() == {"message": "Gradient Nova AI backend running"}
        assert client.get("/docs").status_code == 200
        assert "post" in client.get("/openapi.json").json()["paths"]["/api/analyze"]


@pytest.mark.parametrize("change", [
    {"company": " "}, {"role": ""}, {"experience": -1},
    {"experience": 1.5}, {"experience": True}, {"experience": "ten"},
])
def test_invalid_input(change):
    with TestClient(app) as client:
        response = client.post("/api/analyze", json={
            "company": "Microsoft", "role": "Engineer", "experience": 10, **change,
        })
        assert response.status_code == 422


def test_pagination_unique_ids_comments_and_saved_raw(tmp_path):
    def respond(request):
        page = int(request.url.params["page"])
        kind = request.url.params["tags"]
        hit = {"objectID": str(page * 2 + (1 if kind == "story" else 2)),
               "title": "Microsoft" if kind == "story" else None,
               "comment_text": "Original <p>comment</p>" if kind == "comment" else None}
        return httpx.Response(200, json={"hits": [hit], "nbPages": 2})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = hn.collect("Microsoft", 20, client=client)
    assert result["item_count"] == 4
    assert result["counts"] == {"story": 2, "comment": 2}
    assert result["stop_reason"] == "search_exhausted"
    assert result["requests_made"] == 20
    assert len(result["items"][0]["matched_queries"]) == 5
    assert result["items"][1]["raw"]["comment_text"] == "Original <p>comment</p>"
    assert result["items"][1]["source_url"].endswith("id=2")
    path = hn.save_raw(result, tmp_path)
    assert path.name == "hacker_news_microsoft.json"
    assert json.loads(path.read_text(encoding="utf-8")) == result


def test_limit_and_empty_results():
    def respond(request):
        kind = request.url.params["tags"]
        hits = [{"objectID": str(i + (100 if kind == "comment" else 0)),
                 "title": "Story", "comment_text": "Comment"} for i in range(10)]
        return httpx.Response(200, json={"hits": hits, "nbPages": 10})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = hn.collect("Microsoft", 5, client=client)
    assert result["item_count"] == 5
    assert result["counts"]["comment"] > 0
    assert result["stop_reason"] == "limit_reached"
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"hits": [], "nbPages": 0})
    )) as client:
        result = hn.collect("No matches", client=client)
    assert result["item_count"] == 0
    assert result["stop_reason"] == "search_exhausted"


def test_temporary_error_retry_and_permanent_failure(monkeypatch):
    monkeypatch.setattr(hn.time, "sleep", lambda _: None)
    statuses = iter([429, 503, 200])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(
        next(statuses), json={"hits": [], "nbPages": 0}
    ))) as client:
        assert hn.search_page(client, "Microsoft", "story", 0)["hits"] == []
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(403)
    )) as client:
        with pytest.raises(httpx.HTTPStatusError):
            hn.collect("Microsoft", client=client)


def test_malformed_response():
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"error": "bad response"})
    )) as client:
        with pytest.raises(ValueError, match="Unexpected"):
            hn.collect("Microsoft", client=client)


def test_failed_serialization_preserves_previous_file(tmp_path):
    path = tmp_path / "hacker_news_microsoft.json"
    path.write_text("previous result", encoding="utf-8")
    with pytest.raises(TypeError):
        hn.save_raw({"company": "Microsoft", "invalid": object()}, tmp_path)
    assert path.read_text(encoding="utf-8") == "previous result"
