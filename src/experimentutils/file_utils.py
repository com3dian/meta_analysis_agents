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
    "get_markdown_paths_from_directory", 
    "get_paper_info_from_path",
    "list_paper_folders",
    "get_paper_count",
    "filter_paths_by_pattern",
    "DEFAULT_PAPER_OUTPUT_DIR",
    "DEFAULT_SUBFOLDER",
]
