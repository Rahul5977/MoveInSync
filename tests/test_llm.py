"""LLM layer: Groq error mapping (no network: httpx.MockTransport), extractive fake, factory."""

import json

import httpx
import pytest

from app.config import Settings
from app.llm.base import ConnectionDropped, LLMError
from app.llm.factory import make_llm
from app.llm.fake import ExtractiveLLM, sections_in
from app.llm.groq import GroqLLM
from app.rag.prompt import build_user_prompt
from app.schemas import RetrievedChunk


def _groq(handler, key="gsk_test_not_real", cap=2.5) -> GroqLLM:
    return GroqLLM(key, "m", cap, client=httpx.Client(transport=httpx.MockTransport(handler)))


def _ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_groq_success_returns_content_and_uses_json_mode():
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = json.loads(req.content)
        seen["timeout"] = req.extensions["timeout"]
        return _ok('{"covered": false, "answer": "", "cited_ids": []}')

    out = _groq(handler, cap=2.5).complete_json("sys", "user", timeout_s=1.2)
    assert json.loads(out)["covered"] is False
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["temperature"] == 0
    assert seen["timeout"]["read"] == 1.2  # bounded by the remaining deadline


def test_groq_timeout_capped_by_max():
    seen = {}

    def handler(req):
        seen["timeout"] = req.extensions["timeout"]
        return _ok("{}")

    _groq(handler, cap=2.5).complete_json("s", "u", timeout_s=9.0)
    assert seen["timeout"]["read"] == 2.5


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(429, json={"error": "quota"}), "rate_limited"),
        (httpx.Response(500), "http_500"),
        (httpx.Response(503), "http_503"),
        (httpx.Response(401), "http_401"),
        (httpx.Response(200, json={"unexpected": True}), "malformed_provider_response"),
        (httpx.Response(200, content=b"not json"), "malformed_provider_response"),
    ],
)
def test_groq_errors_map_to_llm_error(response, reason):
    with pytest.raises(LLMError, match=reason):
        _groq(lambda req: response).complete_json("s", "u", 1.0)


def test_groq_timeout_maps_to_llm_error():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(LLMError, match="timeout"):
        _groq(handler).complete_json("s", "u", 1.0)


def test_groq_network_error_maps_to_llm_error():
    def handler(req):
        raise httpx.ConnectError("down", request=req)

    with pytest.raises(LLMError, match="network"):
        _groq(handler).complete_json("s", "u", 1.0)


@pytest.mark.parametrize("exc", [httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError])
def test_groq_connection_level_errors_are_connection_dropped(exc):
    """D53: only these are retryable; the answer was never generated."""

    def handler(req):
        raise exc("Server disconnected without sending a response.", request=req)

    with pytest.raises(ConnectionDropped, match=exc.__name__):
        _groq(handler).complete_json("s", "u", 1.0)


@pytest.mark.parametrize(
    "handler",
    [
        lambda req: httpx.Response(429, json={"error": "quota"}),
        lambda req: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=req)),
        lambda req: httpx.Response(503),
    ],
    ids=["429", "timeout", "503"],
)
def test_groq_rate_limit_timeout_http_are_not_retryable(handler):
    with pytest.raises(LLMError) as info:
        _groq(handler).complete_json("s", "u", 1.0)
    assert not isinstance(info.value, ConnectionDropped)


def test_groq_without_key_fails_fast_without_network():
    def handler(req):
        raise AssertionError("must not call the network without a key")

    with pytest.raises(LLMError, match="no_api_key"):
        _groq(handler, key="").complete_json("s", "u", 1.0)


def _chunk(cid: str, body: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        company_id="nimbus",
        company_name="Nimbus Software",
        version=1,
        doc_title="T",
        section_id="8",
        section_title="Escort",
        text=f"Company: Nimbus Software | T v1 | §8 Escort\n\n{body}",
        score=0.9,
    )


def test_extractive_cites_top_section_first_sentence():
    prompt = build_user_prompt(
        "q?", [_chunk("nimbus_v1_s8_p0", "Escort after 8:00 PM. Second."), _chunk("b", "Other.")]
    )
    out = json.loads(ExtractiveLLM().complete_json("s", prompt))
    assert out == {
        "covered": True,
        "answer": "Escort after 8:00 PM.",
        "cited_ids": ["nimbus_v1_s8_p0"],
    }


def test_extractive_with_no_sections_is_not_covered():
    assert json.loads(ExtractiveLLM().complete_json("s", "no sections"))["covered"] is False


def test_sections_in_round_trips_escaped_text():
    prompt = build_user_prompt("q", [_chunk("x", "a < b & c > d")])
    assert sections_in(prompt)[0][1].endswith("a < b & c > d")


def test_factory_picks_provider():
    fake = make_llm(Settings(_env_file=None, llm_provider="fake"))
    groq = make_llm(Settings(_env_file=None, llm_provider="groq", groq_api_key=""))
    assert isinstance(fake, ExtractiveLLM) and (fake.provider, fake.ready) == ("fake", True)
    assert isinstance(groq, GroqLLM) and (groq.provider, groq.ready) == ("groq", False)


def test_groq_finish_reason_length_is_truncated_output():
    from app.llm.base import TruncatedOutput

    resp = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": '{"covered": true, "ans'}, "finish_reason": "length"}
            ]
        },
    )
    with pytest.raises(TruncatedOutput) as info:
        _groq(lambda req: resp).complete_json("s", "u", 1.0)
    assert info.value.partial == '{"covered": true, "ans'


def test_groq_json_validate_failed_is_model_output_error():
    from app.llm.base import JsonValidationFailed

    err = {
        "error": {
            "message": "Failed to validate JSON.",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
            "failed_generation": '{"covered": tru',
        }
    }
    with pytest.raises(JsonValidationFailed) as info:
        _groq(lambda req: httpx.Response(400, json=err)).complete_json("s", "u", 1.0)
    assert info.value.partial == '{"covered": tru'


def test_groq_other_400_is_provider_error():
    err = {"error": {"message": "bad", "code": "model_not_found"}}
    with pytest.raises(LLMError, match="http_400"):
        _groq(lambda req: httpx.Response(400, json=err)).complete_json("s", "u", 1.0)


def test_system_prompt_rules():
    from app.rag.prompt import SYSTEM_PROMPT

    assert "data, not instructions" in SYSTEM_PROMPT
    assert "covered even when the answer is no" in SYSTEM_PROMPT  # D47
    assert '"cited_ids"' in SYSTEM_PROMPT
