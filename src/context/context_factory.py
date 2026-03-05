"""
ContextFactory - Unified entry point for creating execution contexts.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base_context import ContextType, ExecutionContext
from .document_context import DocumentContext


class ContextFactory:
    """
    Factory for creating ExecutionContext instances with automatic type detection.
    """
    
    EXTENSION_MAP = {
        ".pdf": ContextType.PDF,
        ".md": ContextType.TEXT,
        ".txt": ContextType.TEXT,
    }
    
    @classmethod
    def create(
        cls,
        source: Union[str, List[str], Dict[str, str], ExecutionContext],
        name: str = "context",
        description: Optional[str] = None,
        **kwargs
    ) -> ExecutionContext:
        """
        Create an ExecutionContext from various input formats.
        """
        if isinstance(source, ExecutionContext):
            return source
        
        if isinstance(source, str):
            return cls._create_from_string(source, name, description, **kwargs)
        
        if isinstance(source, list):
            return cls._create_from_list(source, name, description, **kwargs)
        
        if isinstance(source, dict):
            return cls._create_from_dict(source, name, description, **kwargs)
        
        raise ValueError(
            f"Cannot create ExecutionContext from type: {type(source)}."
        )
    
    @classmethod
    def _create_from_string(
        cls,
        path: str,
        name: str,
        description: Optional[str],
        **kwargs
    ) -> ExecutionContext:
        """Create ExecutionContext from a string path."""
        path = os.path.expanduser(path)
        
        if os.path.isdir(path):
            return cls._create_from_directory(path, name, description, **kwargs)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        context_type = cls._detect_type_from_extension(path)
        
        return cls._create_typed_context(
            context_type, 
            path, 
            name, 
            description, 
            **kwargs
        )
    
    @classmethod
    def _create_from_list(
        cls,
        paths: List[str],
        name: str,
        description: Optional[str],
        **kwargs
    ) -> ExecutionContext:
        """Create ExecutionContext from a list of document file paths."""
        if not paths:
            raise ValueError("Empty path list provided")
        
        expanded_paths = []
        for p in paths:
            p = os.path.expanduser(p)
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")
            expanded_paths.append(p)
        
        context_type = cls._detect_type_from_extension(expanded_paths[0])

        if context_type not in (ContextType.TEXT, ContextType.PDF):
            raise ValueError(
                f"List of {context_type.value} files not supported. "
                "Only text/markdown/PDF documents are supported."
            )

        resources = {Path(p).stem: p for p in expanded_paths}
        return DocumentContext(
            resources,
            name=name,
            description=description,
            context_type=context_type,
            **kwargs,
        )
    
    @classmethod
    def _create_from_dict(
        cls,
        resources: Dict[str, str],
        name: str,
        description: Optional[str],
        **kwargs
    ) -> ExecutionContext:
        """Create ExecutionContext from a dict of resource_name -> document path."""
        if not resources:
            raise ValueError("Empty resources dict provided")
        
        for resource_name, path in resources.items():
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found for resource '{resource_name}': {path}")
        
        first_path = list(resources.values())[0]
        context_type = cls._detect_type_from_extension(first_path)

        if context_type not in (ContextType.TEXT, ContextType.PDF):
            raise ValueError(
                f"Dict of {context_type.value} files not supported as multi-resource "
                "document context. Only text/markdown/PDF documents are supported."
            )

        return DocumentContext(
            resources,
            name=name,
            description=description,
            context_type=context_type,
            **kwargs,
        )

    @classmethod
    def _create_from_directory(
        cls,
        dir_path: str,
        name: str,
        description: Optional[str],
        **kwargs,
    ) -> ExecutionContext:
        """
        Create ExecutionContext from a directory of document files.

        This collects all files in the directory whose extension is supported
        by `EXTENSION_MAP` (e.g., .md, .txt, .pdf) and delegates to
        `_create_from_list`.
        """
        dir_path = os.path.expanduser(dir_path)

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        all_entries = [
            os.path.join(dir_path, entry)
            for entry in os.listdir(dir_path)
        ]
        file_paths = [
            p for p in all_entries
            if os.path.isfile(p) and Path(p).suffix.lower() in cls.EXTENSION_MAP
        ]

        if not file_paths:
            supported = ", ".join(sorted(cls.EXTENSION_MAP.keys()))
            raise FileNotFoundError(
                f"No supported document files ({supported}) found in directory: {dir_path}"
            )

        return cls._create_from_list(
            file_paths,
            name=name,
            description=description,
            **kwargs,
        )

    @classmethod
    def _detect_type_from_extension(cls, path: str) -> ContextType:
        """Detect context type from file extension."""
        ext = Path(path).suffix.lower()
        return cls.EXTENSION_MAP.get(ext, ContextType.UNKNOWN)
    
    @classmethod
    def _create_typed_context(
        cls,
        context_type: ContextType,
        path: str,
        name: str,
        description: Optional[str],
        **kwargs
    ) -> ExecutionContext:
        """Create a specific ExecutionContext type for a single path."""
        resource_name = Path(path).stem

        if context_type in (ContextType.TEXT, ContextType.PDF):
            return DocumentContext(
                {resource_name: path},
                name=name,
                description=description,
                context_type=context_type,
                **kwargs,
            )

        raise ValueError(
            f"Unsupported context type '{context_type.value}' for path '{path}'. "
            "Only text/markdown/PDF documents are supported."
        )

# Convenience function
def create_context(
    source: Union[str, List[str], Dict[str, str], ExecutionContext],
    name: str = "context",
    **kwargs
) -> ExecutionContext:
    """
    Convenience function to create an ExecutionContext.
    """
    return ContextFactory.create(source, name=name, **kwargs)
