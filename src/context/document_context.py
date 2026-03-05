"""
DocumentContext - ExecutionContext implementation for text and PDF documents.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from pypdf import PdfReader

from .base_context import (
    ContextType,
    ExecutionContext,
    FieldInfo,
    ResourceInfo,
)


class DocumentContext(ExecutionContext):
    """
    ExecutionContext for collections of unstructured documents (Markdown, text, PDF).

    Each file is exposed as a separate resource.

    - `read_resource` returns either a single string (full content) or a list of
      strings (e.g. paragraphs) depending on the `as_list` kwarg.
    - `iter_resource` yields the same object as `read_resource`, once.
    """

    def __init__(
        self,
        resources: Dict[str, str],
        name: str = "context",
        description: Optional[str] = None,
        context_type: ContextType = ContextType.TEXT,
    ):
        super().__init__(name=name, description=description)
        self._resources = resources
        self._context_type = context_type

    @property
    def context_type(self) -> ContextType:
        return self._context_type

    @property
    def resources(self) -> List[str]:
        return list(self._resources.keys())

    def _load_resource_info(self, resource: str) -> ResourceInfo:
        if resource not in self._resources:
            raise ValueError(f"Unknown resource '{resource}'")

        path = self._resources[resource]
        file_path = Path(path)
        size = file_path.stat().st_size if file_path.exists() else None

        # Single 'content' field for the full text
        field = FieldInfo(
            name="content",
            dtype="string",
            nullable=False,
            description="Full extracted text content of the document.",
        )

        return ResourceInfo(
            name=resource,
            item_count=1,
            field_count=1,
            fields=[field],
            primary_key=None,
            location=str(file_path),
            size_in_bytes=size,
            description=f"Document resource from file '{file_path.name}'",
        )

    def _read_text_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _read_pdf_file(self, path: str) -> str:
        # TODO: use MinerU to read the pdf files
        reader = PdfReader(path)
        parts: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def read_resource(
        self,
        resource: str,
        fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Read a document resource.

        Args:
            resource: Logical resource name (key into `self._resources`).
            fields: Ignored; kept for interface compatibility.
            limit: Ignored; content is always returned in full.
            **kwargs:
                as_list (bool): If True, return a list of string chunks
                    (paragraphs); otherwise return a single string.
        """
        if resource not in self._resources:
            raise ValueError(f"Unknown resource '{resource}'")

        path = self._resources[resource]
        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            content = self._read_pdf_file(path)
        else:
            content = self._read_text_file(path)

        as_list = bool(kwargs.get("as_list", False))
        if as_list:
            # Simple paragraph split; callers can re-chunk as needed.
            return [p for p in content.split("\n\n") if p.strip()]
        return content

    def iter_resource(
        self, resource: str, chunksize: int = 10000, **kwargs
    ) -> Iterator[Union[str, List[str]]]:
        """
        Iterate over a document resource.

        For now, this just yields the same object as `read_resource` once.
        """
        yield self.read_resource(resource, **kwargs)

