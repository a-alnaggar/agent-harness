"""Long-term memory subsystem — persists user facts, FAQs, and session profiles."""

from .core_profile import ALLOWED_CORE_KEYS, CoreProfileManager, is_core_key
from .routing import route_memory_mutation
from .privacy import validate_safe_content
from .sqlite_store import (
    CourseFaqRepository,
    FaqEntry,
    GeneralMemoryEntry,
    GeneralMemoryRepository,
    SqliteMemoryStore,
)
from .tools import (
    ForgetMemoryTool,
    RecallMemoryTool,
    SaveFaqTool,
    SaveMemoryTool,
    SearchFaqTool,
    UpdateCorePreferenceTool,
)

__all__ = [
    "CoreProfileManager",
    "CourseFaqRepository",
    "ALLOWED_CORE_KEYS",
    "FaqEntry",
    "ForgetMemoryTool",
    "GeneralMemoryEntry",
    "GeneralMemoryRepository",
    "RecallMemoryTool",
    "SaveFaqTool",
    "SaveMemoryTool",
    "SearchFaqTool",
    "SqliteMemoryStore",
    "UpdateCorePreferenceTool",
    "is_core_key",
    "route_memory_mutation",
    "validate_safe_content",
]
