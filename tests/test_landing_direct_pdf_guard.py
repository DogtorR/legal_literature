from lit_agent.landing_page import resolve_landing_page


def test_pdf_content_type_response_is_blocked_before_body_read():
    class PdfResponse:
        status_code = 200
        url = "https://publisher.test/article"
        headers = {"Content-Type": "application/pdf"}

        @property
        def text(self):
            raise AssertionError("PDF body should not be read")

        @property
        def content(self):
            raise AssertionError("PDF body should not be read")

    class FakeSession:
        def get(self, *args, **kwargs):
            return PdfResponse()

    result = resolve_landing_page({"doi": "10.1234/a", "landing_url": "https://publisher.test/article"}, dry_run=False, allow_network=True, session=FakeSession())

    assert result.status == "blocked"
    assert result.reason == "doi_landing_pdf_response_not_fetched"


def test_pdf_content_disposition_response_is_blocked_before_body_read():
    class PdfAttachmentResponse:
        status_code = 200
        url = "https://publisher.test/article"
        headers = {"Content-Type": "text/html", "Content-Disposition": "attachment; filename=article.pdf"}

        @property
        def text(self):
            raise AssertionError("PDF attachment body should not be read")

    class FakeSession:
        def get(self, *args, **kwargs):
            return PdfAttachmentResponse()

    result = resolve_landing_page({"doi": "10.1234/a", "landing_url": "https://publisher.test/article"}, dry_run=False, allow_network=True, session=FakeSession())

    assert result.status == "blocked"
    assert result.reason == "doi_landing_pdf_response_not_fetched"
