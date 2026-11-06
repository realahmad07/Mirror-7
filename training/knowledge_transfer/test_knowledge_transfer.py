"""
Unit and integration tests for the Knowledge Transfer Subsystem.
"""
import pytest
from training.knowledge_transfer import (
    KnowledgeCategory,
    ALL_CATEGORIES,
    TeacherProvider,
    TeacherResponse,
    MockTeacher,
    TaskGenerator,
    TaskPrompt,
    KnowledgeCollector,
    ResponseValidator,
    ValidationResult,
    DatasetNormalizer,
    DistillationManager,
)
from training.schema import TrainingExample


def test_categories_registry_completeness():
    assert len(ALL_CATEGORIES) == 13
    assert KnowledgeCategory.GENERAL_CONVERSATION in ALL_CATEGORIES
    assert KnowledgeCategory.MATHEMATICS in ALL_CATEGORIES
    assert KnowledgeCategory.LOGICAL_REASONING in ALL_CATEGORIES
    assert KnowledgeCategory.PROGRAMMING_AND_DEBUGGING in ALL_CATEGORIES
    assert KnowledgeCategory.SELF_CORRECTION in ALL_CATEGORIES
    assert KnowledgeCategory.MULTI_STEP_PROBLEM_SOLVING in ALL_CATEGORIES


def test_task_generator_seeds_and_corpus():
    gen = TaskGenerator(seed=123)
    seeds = gen.generate_seed_tasks()
    assert len(seeds) >= 13 * 3  # at least 3 seeds per category
    
    # Check that all categories are covered in seeds
    seed_cats = {s.category for s in seeds}
    assert seed_cats == set(ALL_CATEGORIES)

    # Test corpus generation
    corpus = gen.generate_corpus(target_per_category=4)
    assert len(corpus) == 13 * 4
    
    # Test variation generation
    variations = gen.generate_variations(seeds[0], count=2)
    assert len(variations) == 2
    assert variations[0].task_id != seeds[0].task_id


def test_mock_teacher_response():
    teacher = MockTeacher(model_id="mock-qwen-7b")
    resp = teacher.generate("Calculate 2+2", KnowledgeCategory.MATHEMATICS)
    assert isinstance(resp, TeacherResponse)
    assert resp.category == KnowledgeCategory.MATHEMATICS
    assert resp.model_id == "mock-qwen-7b"
    assert "Step" in resp.response_text
    assert resp.latency_seconds >= 0.0
    resp.validate()


def test_response_validator():
    validator = ResponseValidator(min_bytes=10, max_bytes=2000)

    # Valid response
    valid_resp = TeacherResponse(
        prompt="Write a function",
        response_text="```python\ndef add(a, b):\n    return a + b\n```\nExplanation: Simple function.",
        category=KnowledgeCategory.PROGRAMMING_AND_DEBUGGING,
        model_id="test-model",
        provider="test",
    )
    res = validator.validate(valid_resp)
    assert res.is_valid is True
    assert res.quality_score >= 0.7

    # Invalid code syntax
    bad_code_resp = TeacherResponse(
        prompt="Write a function",
        response_text="```python\ndef broken(\n    invalid syntax\n```",
        category=KnowledgeCategory.PROGRAMMING_AND_DEBUGGING,
        model_id="test-model",
        provider="test",
    )
    res_bad = validator.validate(bad_code_resp)
    assert "syntax validation" in str(res_bad.reasons)

    # Repetition failure
    looping_text = "word " * 50
    looping_resp = TeacherResponse(
        prompt="Say word",
        response_text=looping_text,
        category=KnowledgeCategory.GENERAL_CONVERSATION,
        model_id="test-model",
        provider="test",
    )
    res_loop = validator.validate(looping_resp)
    assert res_loop.is_valid is False
    assert any("repetition" in r.lower() for r in res_loop.reasons)


def test_dataset_normalizer_and_provenance(tmp_path):
    normalizer = DatasetNormalizer(default_split="train")
    teacher = MockTeacher()
    resp = teacher.generate("How does sunlight work?", KnowledgeCategory.FACTUAL_EXPLANATION)
    val = ValidationResult(is_valid=True, quality_score=0.95, metrics={"test": 1})

    example = normalizer.to_training_example("task_001", resp, val)
    assert isinstance(example, TrainingExample)
    assert example.example_id == "kt_task_001"
    assert example.quality == 0.95
    assert example.state["teacher_model"] == teacher.model_id
    assert example.state["category"] == "factual_explanation"
    assert example.split == "train"

    # Export test
    out_file = tmp_path / "normalized_test.jsonl"
    count = normalizer.export_jsonl([example], out_file)
    assert count == 1
    assert out_file.exists()


def test_knowledge_collector_pipeline():
    teacher = MockTeacher()
    collector = KnowledgeCollector(teacher=teacher)
    tasks = [
        TaskPrompt("t1", KnowledgeCategory.MATHEMATICS, "Compute 10 * 5"),
        TaskPrompt("t2", KnowledgeCategory.PROGRAMMING_AND_DEBUGGING, "Write hello world in python"),
    ]

    examples, report = collector.collect(tasks)
    assert len(examples) == 2
    assert report.total_tasks == 2
    assert report.valid_examples == 2
    assert report.rejected_examples == 0
    assert "mathematics" in report.examples_by_category
    assert "programming_and_debugging" in report.examples_by_category


def test_distillation_manager():
    dm = DistillationManager()
    info = dm.explain_vocabulary_alignment_constraint()
    assert "INCOMPATIBLE" in info["logit_distillation_status"]

    # Test demonstration filtering
    normalizer = DatasetNormalizer()
    teacher = MockTeacher()
    resp = teacher.generate("Prompt 1", KnowledgeCategory.GENERAL_CONVERSATION)
    ex1 = normalizer.to_training_example("1", resp, ValidationResult(True, 0.9))
    ex2 = normalizer.to_training_example("2", resp, ValidationResult(True, 0.4))
    
    demos = dm.build_demonstration_dataset([ex1, ex2])
    assert len(demos) == 1
    assert demos[0].example_id == ex1.example_id
