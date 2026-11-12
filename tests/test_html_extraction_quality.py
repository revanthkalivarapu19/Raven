from datetime import datetime

from rag_ingestion.extractor.html_cleaner import HTMLCleaner


def test_html_cleaner_prefers_article_and_keeps_source_authored_descriptive_alt():
    alt = (
        "A graphic shows the Sun at the center with Earth's orbit drawn as a blue circle around it. "
        "Earth appears along the orbit to the right of the Sun, at the 3 o'clock position on the circle."
    )
    html = f"""
    <html><head><title>Interstellar Mapping and Acceleration Probe (IMAP)</title></head>
      <body><header>NASA global navigation</header>
        <main><nav>Exoplanets topics</nav><article>
          <p>Mission overview.</p><img alt="{alt}">
        </article></main><footer>Site links</footer>
      </body>
    </html>
    """
    doc = HTMLCleaner().extract_from_content(
        html.encode("utf-8"),
        {
            "source": "NASA",
            "domain": "science",
            "url": "https://science.nasa.gov/mission/imap/",
            "retrieved_date": datetime(2026, 9, 27),
        },
    )

    assert doc is not None
    assert "Mission overview." in doc.text
    assert "Image description: " + alt in doc.text
    assert "global navigation" not in doc.text
    assert "Exoplanets topics" not in doc.text
    assert doc.source == "NASA"
    assert doc.url == "https://science.nasa.gov/mission/imap/"
    assert doc.retrieved_date == datetime(2026, 9, 27)


def test_html_cleaner_rejects_metadata_only_or_navigation_only_pages():
    html = """<html><head><title>News</title></head><body><main>
      <p>Skip to main content Subscribe Log In Cookie preferences</p>
    </main></body></html>"""
    assert HTMLCleaner().extract_from_content(
        html.encode("utf-8"), {"source": "Example", "domain": "general"}
    ) is None
