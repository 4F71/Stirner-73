"""web_ops hata yolu testleri — network mock'lu."""

import requests
from unittest.mock import MagicMock, patch

from tools.web_ops import fetch_url, whois_lookup, web_search


# ── fetch_url ────────────────────────────────────────────────────────────────

def test_fetch_url_404_returns_error():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError(
        response=MagicMock(status_code=404)
    )
    with patch("tools.web_ops.requests.get", return_value=mock_resp):
        result = fetch_url("https://example.com/notfound")
    assert "hata" in result.lower() or "error" in result.lower()


def test_fetch_url_timeout_returns_error():
    with patch("tools.web_ops.requests.get", side_effect=requests.Timeout):
        result = fetch_url("https://slow.example.com")
    assert "hata" in result.lower() or "timeout" in result.lower() or "error" in result.lower()


def test_fetch_url_connection_error_returns_error():
    with patch("tools.web_ops.requests.get", side_effect=requests.ConnectionError):
        result = fetch_url("https://unreachable.example.com")
    assert "hata" in result.lower() or "error" in result.lower()


# ── web_search ───────────────────────────────────────────────────────────────

def test_web_search_exception_returns_error_string():
    with patch("tools.web_ops.DDGS") as mock_ddgs_cls:
        mock_ddgs_cls.return_value.__enter__.side_effect = Exception("network down")
        result = web_search("python örneği")
    assert "hata" in result.lower() or "error" in result.lower()


def test_web_search_empty_results():
    with patch("tools.web_ops.DDGS") as mock_ddgs_cls:
        mock_ctx = MagicMock()
        mock_ctx.text.return_value = []
        mock_ddgs_cls.return_value.__enter__.return_value = mock_ctx
        result = web_search("asdflkjqwer12345xnyz")
    assert "bulunamadı" in result.lower() or "sonuç" in result.lower()


# ── whois_lookup ─────────────────────────────────────────────────────────────

def test_whois_lookup_connection_error():
    with patch("tools.web_ops.requests.get", side_effect=requests.ConnectionError):
        result = whois_lookup("example.com")
    assert "hata" in result.lower() or "error" in result.lower()


def test_whois_lookup_timeout():
    with patch("tools.web_ops.requests.get", side_effect=requests.Timeout):
        result = whois_lookup("example.com")
    assert "hata" in result.lower() or "timeout" in result.lower() or "error" in result.lower()
