"""
Mirror 7 Knowledge Transfer Subsystem.

Provides tools for transferring structured knowledge and reasoning demonstrations
from capable teacher models into Mirror 7's native student architecture without
creating a permanent production dependency.
"""

from .categories import KnowledgeCategory, ALL_CATEGORIES
from .teacher_interface import TeacherProvider, TeacherResponse, MockTeacher
from .task_generator import TaskGenerator, TaskPrompt
from .collector import KnowledgeCollector
from .validator import ResponseValidator, ValidationResult
from .normalizer import DatasetNormalizer
from .distillation import DistillationManager

__all__ = [
    "KnowledgeCategory",
    "ALL_CATEGORIES",
    "TeacherProvider",
    "TeacherResponse",
    "MockTeacher",
    "TaskGenerator",
    "TaskPrompt",
    "KnowledgeCollector",
    "ResponseValidator",
    "ValidationResult",
    "DatasetNormalizer",
    "DistillationManager",
]
