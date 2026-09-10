"""数据源解析纯函数测试（离线，不联网）。"""

import json

from paper_agent.sources.arxiv import parse_atom
from paper_agent.sources.europepmc import parse_hit
from paper_agent.sources.openalex import rebuild_abstract


def test_openalex_rebuild_abstract():
    inv = {"world": [1], "Hello": [0]}
    assert rebuild_abstract(inv) == "Hello world"
    assert rebuild_abstract({}) == ""


def test_europepmc_parse_hit():
    hit = {
        "id": "36626471",
        "source": "MED",
        "pmcid": "PMC9860475",
        "title": "A <b>bold</b> study ",
        "authorString": "Smith J, Doe A",
        "pubYear": "2023",
        "citedByCount": 5,
        "doi": "10.1/x",
        "abstractText": "Abs.",
        "fullTextUrlList": {
            "fullTextUrl": [
                {"availability": "Open access", "documentStyle": "pdf",
                 "url": "https://europepmc.org/articles/PMC9860475?pdf=render"},
                {"availability": "Open access", "documentStyle": "html",
                 "url": "https://europepmc.org/articles/PMC9860475"},
            ]
        },
    }
    p = parse_hit(hit)
    assert p.source_id == "europepmc:PMC9860475"
    assert p.title == "A bold study"  # HTML 标签被剥离
    assert p.year == 2023
    assert p.pdf_urls == ["https://europepmc.org/articles/PMC9860475?pdf=render"]
    assert not p.abstract_only


def test_europepmc_parse_hit_abstract_only():
    hit = {"id": "123", "source": "MED", "title": "T", "pubYear": None,
           "fullTextUrlList": {"fullTextUrl": []}}
    p = parse_hit(hit)
    assert p.source_id == "europepmc:MED:123"
    assert p.abstract_only and p.pdf_urls == []


_ARXML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>  Attention
 Is All You Need </title>
    <summary>Abstract text.</summary>
    <published>2024-01-20T17:59:59Z</published>
    <author><name>A. Author</name></author>
    <author><name>B. Author</name></author>
    <link href="http://arxiv.org/pdf/2401.12345v1" type="application/pdf" rel="related"/>
  </entry>
</feed>
"""


def test_arxiv_parse_atom():
    papers = parse_atom(_ARXML)
    assert len(papers) == 1
    p = papers[0]
    assert p.source_id == "arxiv:2401.12345"  # 版本号已剥离
    assert p.title == "Attention Is All You Need"  # 换行空白已压缩
    assert p.year == 2024
    assert p.authors == ["A. Author", "B. Author"]
    assert p.pdf_urls == ["http://arxiv.org/pdf/2401.12345v1"]


def test_arxiv_parse_atom_no_link_falls_back():
    xml = _ARXML.replace(
        '<link href="http://arxiv.org/pdf/2401.12345v1" type="application/pdf" rel="related"/>',
        "",
    )
    p = parse_atom(xml)[0]
    assert p.pdf_urls == ["https://arxiv.org/pdf/2401.12345"]
