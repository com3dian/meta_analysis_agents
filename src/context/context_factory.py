"""
ContextFactory - Unified entry point for creating execution contexts.
"""
import os
from typing import Union, List, Dict, Optional, Any
from pathlib import Path
import glob

from .base_context import ExecutionContext, ContextType
from .csv_context import CSVContext
from .sqlite_context import SQLiteContext


class ContextFactory:
    """
    Factory for creating ExecutionContext instances with automatic type detection.
    """
    
    EXTENSION_MAP = {
        '.csv': ContextType.CSV,
        '.tsv': ContextType.CSV,
        '.txt': ContextType.CSV,
        '.sqlite': ContextType.SQLITE,
        '.sqlite3': ContextType.SQLITE,
        '.db': ContextType.SQLITE,
        '.parquet': ContextType.PARQUET,
        '.json': ContextType.JSON,
        '.jsonl': ContextType.JSON,
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
        """Create ExecutionContext from a list of file paths."""
        if not paths:
            raise ValueError("Empty path list provided")
        
        expanded_paths = []
        for p in paths:
            p = os.path.expanduser(p)
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")
            expanded_paths.append(p)
        
        context_type = cls._detect_type_from_extension(expanded_paths[0])
        
        if context_type == ContextType.CSV:
            resources = {Path(p).stem: p for p in expanded_paths}
            return CSVContext(resources, name=name, description=description, **kwargs)
        
        raise ValueError(
            f"List of {context_type.value} files not supported."
        )
    
    @classmethod
    def _create_from_dict(
        cls,
        resources: Dict[str, str],
        name: str,
        description: Optional[str],
        **kwargs
    ) -> ExecutionContext:
        """Create ExecutionContext from a dict of resource_name -> path."""
        if not resources:
            raise ValueError("Empty resources dict provided")
        
        for resource_name, path in resources.items():
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found for resource '{resource_name}': {path}")
        
        first_path = list(resources.values())[0]
        context_type = cls._detect_type_from_extension(first_path)
        
        if context_type == ContextType.CSV:
            return CSVContext(resources, name=name, description=description, **kwargs)
        
        raise ValueError(
            f"Dict of {context_type.value} files not supported as multi-resource context."
        )
    
    @classmethod
    def _create_from_directory(
        cls,
        dir_path: str,
        name: str,
        description: Optional[str],
        pattern: str = "*.csv",
        **kwargs
    ) -> ExecutionContext:
        """Create ExecutionContext from a directory of files."""
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        search_pattern = os.path.join(dir_path, pattern)
        files = glob.glob(search_pattern)
        
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {dir_path}"
            )
        
        resources = {Path(f).stem: f for f in files}
        
        if pattern.endswith('.csv') or pattern.endswith('.tsv'):
            return CSVContext(resources, name=name, description=description, **kwargs)
        
        return CSVContext(resources, name=name, description=description, **kwargs)
    
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
        """Create a specific ExecutionContext type."""
        if context_type == ContextType.CSV:
            return CSVContext(path, name=name, description=description, **kwargs)
        
        elif context_type == ContextType.SQLITE:
            return SQLiteContext(path, name=name, description=description, **kwargs)
        
        elif context_type == ContextType.PARQUET:
            raise NotImplementedError("Parquet support coming soon")
        
        elif context_type == ContextType.JSON:
            raise NotImplementedError("JSON support coming soon")
        
        else:
            return CSVContext(path, name=name, description=description, **kwargs)

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
