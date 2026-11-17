"""Mirror 7 Native Intelligence Subsystem (CPU-first).

Provides specialized cognitive organs with verified, trust-gated execution:
* PersonalFactMemory - session-level user fact tracking and recall
* ConversationOrgan - greetings, identity, capabilities, jokes, empathy
* IndependentArithmeticEvaluator & ArithmeticOrgan - exact Fraction arithmetic & AST evaluation
* LogicOrgan - syllogisms, transitive comparisons, propositional reasoning
* WordProblemSolver - schema-based word problem solver
* CuratedKnowledgeOrgan - high-precision canonical knowledge pack
* RetrievalGuard - relevance filter for vault rules
* KnowledgeGraphEngine - SQLite entity_triples engine with indexed sub-millisecond retrieval
* MicroEmbeddingEngine - deterministic subword n-gram vector similarity on CPU
* MultiHopChainingEngine - relational multi-hop graph reasoning
* DocumentAutoCrystallizerEngine - 1-click document parser committing verified triples and rules
"""
from __future__ import annotations

from .types import NativeAnswer
from .memory import PersonalFactMemory
from .chat import ConversationOrgan
from .arithmetic import IndependentArithmeticEvaluator, ArithmeticOrgan, values_agree
from .logic import LogicOrgan
from .word_problems import WordProblemSolver
from .curated_knowledge import CuratedKnowledgeOrgan
from .retrieval_guard import RetrievalGuard
from .knowledge_graph import KnowledgeGraphEngine, FactTriple
from .semantic_search import MicroEmbeddingEngine
from .multi_hop import MultiHopChainingEngine
from .crystallizer import DocumentAutoCrystallizerEngine
from .context_reasoner import ContextReasoner

__all__ = [
    "NativeAnswer",
    "PersonalFactMemory",
    "ConversationOrgan",
    "IndependentArithmeticEvaluator",
    "ArithmeticOrgan",
    "values_agree",
    "LogicOrgan",
    "WordProblemSolver",
    "CuratedKnowledgeOrgan",
    "RetrievalGuard",
    "KnowledgeGraphEngine",
    "FactTriple",
    "MicroEmbeddingEngine",
    "MultiHopChainingEngine",
    "DocumentAutoCrystallizerEngine",
    "ContextReasoner",
]
