"""Standard-library HTML parsing and URL normalization."""
from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit


def canonicalize(url: str) -> str:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in ("http", "https") or not host:
        raise ValueError("Only absolute http(s) URLs are supported")
    port = parts.port
    authority = host if ":" not in host else f"[{host}]"
    if port and not (scheme == "http" and port == 80 or scheme == "https" and port == 443):
        authority += f":{port}"
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, authority, path, query, ""))


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._title = False
        self._ignore = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._ignore += 1
        if tag == "title":
            self._title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._ignore = max(0, self._ignore - 1)
        if tag == "title":
            self._title = False

    def handle_data(self, data: str) -> None:
        if self._ignore:
            return
        value = data.strip()
        if value:
            self.text_parts.append(value)
            if self._title:
                self.title_parts.append(value)


def parse_page(html: str, url: str) -> tuple[str, str, list[str]]:
    parser = PageParser()
    parser.feed(html)
    title = " ".join(parser.title_parts).strip()
    text = " ".join(parser.text_parts)
    links: list[str] = []
    for href in parser.links:
        try:
            links.append(canonicalize(urljoin(url, href)))
        except ValueError:
            continue
    return title, text, links
