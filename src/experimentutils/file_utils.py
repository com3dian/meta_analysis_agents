"""
File utilities for meta-analysis experiments.

This module provides functions for finding and managing paper markdown files
in the experiment data directories.

Directory structure expected:
    data/wopke_100/paper_output/
    ├── 1. Mason 1986 Cassava-cowpea.../
    │   └── hybrid_auto/
    │       ├── 1. Mason 1986 Cassava-cowpea....md  <-- target markdown file
    │       ├── ...other files (json, pdf, images/)
    ├── 2. Paper Name.../
    │   └── hybrid_auto/
    │       ├── 2. Paper Name....md
    │       └── ...
    └── ...
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Default data paths
DEFAULT_PAPER_OUTPUT_DIR = "data/wopke_100/paper_output"
DEFAULT_SUBFOLDER = "hybrid_auto"
DEFAULT_PDF_DIR = "data/wopke_100/papers"


def get_all_markdown_paths(
    base_dir: Optional[str] = None,
    subfolder: str = DEFAULT_SUBFOLDER,
    absolute: bool = True,
) -> List[str]:
    """
    Get all markdown file paths from the paper output directory.
    
    This function scans the paper output directory and returns paths to all
    markdown files found in the expected structure (paper_folder/subfolder/*.md).
    
    Args:
        base_dir: Base directory containing paper folders. 
                  Defaults to "data/wopke_100/paper_output" relative to project root.
        subfolder: Subfolder within each paper folder containing the markdown file.
                   Defaults to "hybrid_auto".
        absolute: If True, return absolute paths. If False, return paths relative
                  to the project root.
    
    Returns:
        List of paths to markdown files, sorted alphabetically by paper folder name.
        
    Example:
        >>> paths = get_all_markdown_paths()
        >>> print(len(paths))
        100
        >>> print(paths[0])
        '/home/user/project/data/wopke_100/paper_output/1. Mason 1986.../hybrid_auto/1. Mason 1986....md'
    """
    # Determine base directory
    if base_dir is None:
        # Try to find project root by looking for common markers
        project_root = _find_project_root()
        base_dir = os.path.join(project_root, DEFAULT_PAPER_OUTPUT_DIR)
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    markdown_paths = []
    
    # Iterate through paper folders
    for paper_folder in sorted(base_path.iterdir()):
        if not paper_folder.is_dir():
            continue
        
        # Look in the subfolder
        subfolder_path = paper_folder / subfolder
        if not subfolder_path.exists():
            continue
        
        # Find markdown files in the subfolder
        for md_file in subfolder_path.glob("*.md"):
            if absolute:
                markdown_paths.append(str(md_file.absolute()))
            else:
                # Make relative to project root
                project_root = _find_project_root()
                try:
                    rel_path = md_file.relative_to(project_root)
                    markdown_paths.append(str(rel_path))
                except ValueError:
                    # If can't make relative, use absolute
                    markdown_paths.append(str(md_file.absolute()))
    
    return markdown_paths


def get_markdown_paths_from_directory(
    directory: str,
    recursive: bool = True,
    absolute: bool = True,
) -> List[str]:
    """
    Get all markdown file paths from a given directory.
    
    This is a more general function that can search any directory for markdown files.
    
    Args:
        directory: Directory to search for markdown files.
        recursive: If True, search recursively in subdirectories.
        absolute: If True, return absolute paths.
        
    Returns:
        List of paths to markdown files found.
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    pattern = "**/*.md" if recursive else "*.md"
    markdown_paths = []
    
    for md_file in sorted(dir_path.glob(pattern)):
        if absolute:
            markdown_paths.append(str(md_file.absolute()))
        else:
            markdown_paths.append(str(md_file))
    
    return markdown_paths


def get_paper_info_from_path(md_path: str) -> Dict[str, str]:
    """
    Extract paper information from a markdown file path.
    
    This function parses the path to extract metadata about the paper.
    
    Args:
        md_path: Path to the markdown file.
        
    Returns:
        Dictionary with keys:
            - 'path': Full path to the markdown file
            - 'filename': Filename without extension
            - 'paper_folder': Name of the paper folder
            - 'paper_name': Extracted paper name (may include number prefix)
            
    Example:
        >>> info = get_paper_info_from_path("/path/to/1. Mason 1986 Cassava.../hybrid_auto/1. Mason....md")
        >>> print(info['paper_folder'])
        '1. Mason 1986 Cassava-cowpea...'
    """
    path = Path(md_path)
    
    # Get filename without extension
    filename = path.stem
    
    # Get parent folder (should be hybrid_auto or similar)
    subfolder = path.parent.name
    
    # Get paper folder (parent of subfolder)
    paper_folder = path.parent.parent.name
    
    return {
        'path': str(path.absolute()),
        'filename': filename,
        'subfolder': subfolder,
        'paper_folder': paper_folder,
        'paper_name': filename,  # Usually same as paper folder name
    }


def list_paper_folders(
    base_dir: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    List all paper folders and their corresponding markdown files.
    
    Args:
        base_dir: Base directory containing paper folders.
                  Defaults to "data/wopke_100/paper_output".
    
    Returns:
        List of tuples (paper_folder_name, markdown_file_path).
        
    Example:
        >>> folders = list_paper_folders()
        >>> for name, path in folders[:3]:
        ...     print(f"{name}: {path}")
    """
    if base_dir is None:
        project_root = _find_project_root()
        base_dir = os.path.join(project_root, DEFAULT_PAPER_OUTPUT_DIR)
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    results = []
    
    for paper_folder in sorted(base_path.iterdir()):
        if not paper_folder.is_dir():
            continue
        
        # Look for markdown in hybrid_auto subfolder
        subfolder_path = paper_folder / DEFAULT_SUBFOLDER
        if not subfolder_path.exists():
            continue
        
        # Find the markdown file
        md_files = list(subfolder_path.glob("*.md"))
        if md_files:
            results.append((paper_folder.name, str(md_files[0].absolute())))
    
    return results


def get_all_paper_paths(
    base_dir: Optional[str] = None,
    pdf_dir: Optional[str] = None,
    subfolder: str = DEFAULT_SUBFOLDER,
    absolute: bool = True,
) -> List[str]:
    """
    Get all paper paths, preferring converted markdown files but falling back
    to raw PDFs when the markdown output directory is empty.

    Priority:
    1. Markdown files in ``{base_dir}/{paper_folder}/{subfolder}/*.md``
       (the pre-converted output from ``data/wopke_100/paper_output/``).
    2. PDF files in ``{pdf_dir}/`` (``data/wopke_100/papers/``) when no
       markdown files are found.

    Args:
        base_dir: Directory containing per-paper sub-folders with markdown
                  output.  Defaults to ``data/wopke_100/paper_output``.
        pdf_dir:  Directory containing the raw PDF files.
                  Defaults to ``data/wopke_100/papers``.
        subfolder: Sub-folder name within each paper folder that holds the
                   ``.md`` file (default: ``"hybrid_auto"``).
        absolute: Return absolute paths when *True* (default).

    Returns:
        List of paper paths sorted by filename.  When markdown files are
        present the list contains ``.md`` paths; otherwise ``.pdf`` paths.

    Example:
        >>> paths = get_all_paper_paths()
        >>> print(paths[0])
        '/…/data/wopke_100/paper_output/1. Mason 1986…/hybrid_auto/1. Mason….md'
        # or if paper_output is empty:
        '/…/data/wopke_100/papers/1. Mason 1986….pdf'
    """
    project_root = _find_project_root()

    # ── Try markdown files first ──────────────────────────────────────────────
    if base_dir is None:
        base_dir = os.path.join(project_root, DEFAULT_PAPER_OUTPUT_DIR)

    md_paths = []
    base_path = Path(base_dir)
    if base_path.exists():
        for paper_folder in sorted(base_path.iterdir()):
            if not paper_folder.is_dir():
                continue
            subfolder_path = paper_folder / subfolder
            if not subfolder_path.exists():
                continue
            for md_file in subfolder_path.glob("*.md"):
                md_paths.append(str(md_file.absolute()) if absolute else str(md_file))

    if md_paths:
        return md_paths

    # ── Fall back to PDF files ────────────────────────────────────────────────
    if pdf_dir is None:
        pdf_dir = os.path.join(project_root, DEFAULT_PDF_DIR)

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Neither markdown output directory ({base_dir}) nor PDF directory "
            f"({pdf_dir}) contain any paper files."
        )

    pdf_paths = sorted(
        str(p.absolute()) if absolute else str(p)
        for p in pdf_path.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
    return pdf_paths


def convert_pdf_to_markdown(
    pdf_path: str,
    output_dir: Optional[str] = None,
    subfolder: str = DEFAULT_SUBFOLDER,
    overwrite: bool = False,
) -> str:
    """
    Convert a PDF paper to a Markdown file saved in the expected
    ``paper_output/{paper_stem}/{subfolder}/{paper_stem}.md`` directory structure.

    The converter uses ``pymupdf`` (PyMuPDF) for text extraction.
    Each PDF page is rendered as a plain-text block separated by a page-break
    comment, which is sufficient for LLM consumption.

    Args:
        pdf_path: Absolute or relative path to the source PDF.
        output_dir: Root directory for output (defaults to
                    ``data/wopke_100/paper_output`` relative to project root).
        subfolder:  Sub-folder within the per-paper directory where the
                    ``.md`` file is written (default: ``"hybrid_auto"``).
        overwrite:  If *True*, overwrite an existing ``.md`` file.
                    If *False* (default), return the existing path immediately.

    Returns:
        Absolute path to the written (or already-existing) Markdown file.

    Raises:
        ImportError: If ``pymupdf`` is not installed.
        FileNotFoundError: If ``pdf_path`` does not exist.
    """
    try:
        import pymupdf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pymupdf is required for PDF-to-Markdown conversion.  "
            "Install it with: pip install pymupdf"
        ) from exc

    pdf_path_obj = Path(pdf_path).resolve()
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    paper_stem = pdf_path_obj.stem  # e.g. "1. Mason 1986 Cassava-cowpea…"

    if output_dir is None:
        project_root = _find_project_root()
        output_dir = os.path.join(project_root, DEFAULT_PAPER_OUTPUT_DIR)

    md_dir  = Path(output_dir) / paper_stem / subfolder
    md_path = md_dir / f"{paper_stem}.md"

    if md_path.exists() and not overwrite:
        return str(md_path)

    md_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path_obj))

    lines: list = [f"# {paper_stem}\n"]
    for page_num, page in enumerate(doc, start=1):
        lines.append(f"\n<!-- page {page_num} -->\n")
        blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1, text, …)
        # Sort top-to-bottom, left-to-right
        blocks = sorted(blocks, key=lambda b: (round(b[1] / 20), b[0]))
        for block in blocks:
            text = block[4].strip() if len(block) > 4 else ""
            if text:
                lines.append(text + "\n")

    doc.close()

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return str(md_path)


def convert_all_pdfs_to_markdown(
    pdf_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    subfolder: str = DEFAULT_SUBFOLDER,
    overwrite: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    Batch-convert all PDF files in ``pdf_dir`` to Markdown.

    Calls :func:`convert_pdf_to_markdown` for every ``.pdf`` file found.
    Already-converted files are skipped unless ``overwrite=True``.

    Args:
        pdf_dir:    Directory containing PDF files
                    (defaults to ``data/wopke_100/papers``).
        output_dir: Root output directory
                    (defaults to ``data/wopke_100/paper_output``).
        subfolder:  Sub-folder name within each per-paper directory.
        overwrite:  Passed through to :func:`convert_pdf_to_markdown`.
        verbose:    Print progress when *True*.

    Returns:
        List of absolute paths to all written (or already-existing) Markdown files.
    """
    project_root = _find_project_root()

    if pdf_dir is None:
        pdf_dir = os.path.join(project_root, DEFAULT_PDF_DIR)

    pdf_files = sorted(
        p for p in Path(pdf_dir).iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )

    md_paths: List[str] = []
    for i, pdf in enumerate(pdf_files, start=1):
        if verbose:
            print(f"[{i}/{len(pdf_files)}] {pdf.name[:70]}")
        md = convert_pdf_to_markdown(
            str(pdf),
            output_dir=output_dir,
            subfolder=subfolder,
            overwrite=overwrite,
        )
        md_paths.append(md)

    if verbose:
        print(f"\nDone — {len(md_paths)} markdown files ready.")
    return md_paths


def read_paper_text(file_path: str) -> str:
    """
    Read the full text of a paper, supporting both Markdown and PDF files.

    For Markdown (``.md``) files the content is returned as-is.
    For PDF files all page text is extracted with ``pypdf`` and joined with
    newlines.

    Args:
        file_path: Path to a ``.md`` or ``.pdf`` file.

    Returns:
        Full text content of the document.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext in (".md", ".markdown", ".txt"):
        return path.read_text(encoding="utf-8")

    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pypdf is required to read PDF files.  Install it with: pip install pypdf"
            ) from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raise ValueError(
        f"Unsupported file extension '{ext}'.  Expected .md, .markdown, .txt, or .pdf."
    )


def _find_project_root() -> str:
    """
    Find the project root directory by looking for common markers.
    
    Returns:
        Path to the project root directory.
    """
    # Start from this file's location and go up
    current = Path(__file__).resolve()
    
    # Look for common project markers
    markers = ['requirements.txt', '.git', 'setup.py', 'pyproject.toml', 'src']
    
    for parent in [current] + list(current.parents):
        for marker in markers:
            if (parent / marker).exists():
                return str(parent)
    
    # Fallback to current working directory
    return os.getcwd()


def get_paper_count(base_dir: Optional[str] = None) -> int:
    """
    Get the total count of papers (markdown files) available.
    
    Args:
        base_dir: Base directory containing paper folders.
        
    Returns:
        Number of markdown files found.
    """
    paths = get_all_markdown_paths(base_dir=base_dir)
    return len(paths)


def filter_paths_by_pattern(
    paths: List[str],
    pattern: str,
    case_sensitive: bool = False,
) -> List[str]:
    """
    Filter markdown paths by a search pattern in the filename.
    
    Args:
        paths: List of markdown file paths.
        pattern: Search pattern to match against filenames.
        case_sensitive: Whether the search should be case-sensitive.
        
    Returns:
        Filtered list of paths matching the pattern.
        
    Example:
        >>> all_paths = get_all_markdown_paths()
        >>> maize_papers = filter_paths_by_pattern(all_paths, "maize")
    """
    if not case_sensitive:
        pattern = pattern.lower()
    
    filtered = []
    for path in paths:
        filename = Path(path).stem
        if not case_sensitive:
            filename = filename.lower()
        if pattern in filename:
            filtered.append(path)
    
    return filtered


# Export all functions
__all__ = [
    "get_all_markdown_paths",
    "get_all_paper_paths",
    "read_paper_text",
    "convert_pdf_to_markdown",
    "convert_all_pdfs_to_markdown",
    "get_markdown_paths_from_directory", 
    "get_paper_info_from_path",
    "list_paper_folders",
    "get_paper_count",
    "filter_paths_by_pattern",
    "DEFAULT_PAPER_OUTPUT_DIR",
    "DEFAULT_SUBFOLDER",
]
