# Repository Organization

## Why Mirror 7 is large

Mirror 7 is a research repository, not a single application package. It contains historical phase implementations, acceptance suites, training systems, backend code, deployment files, and UI assets.

The organization follows a preservation-first approach:

- current engineering entry points are easy to find;
- research provenance is retained;
- phase directories are not flattened;
- old CI definitions are archived instead of deleted;
- long-form explanation lives in docs/.

## Primary areas

| Path | Meaning |
|---|---|
| mirror7_backend/ | Current backend runtime/service package. |
| training/ | Dataset, model, training, evaluation, and Colab assets. |
| ui/ | External presentation layer. |
| phase* | Research phase implementations and their tests. |
| phase355* through phase366* | Response-generation/backend integration line. |
| mirror7_algorithm* | Algorithm/source-family research artifacts. |
| docs/ | Human-facing documentation and archive. |
| .github/workflows/ | Current active CI only. |
| docs/archive/legacy-workflows/ | Inactive historical workflow definitions. |

## Training arrangement

~~~text
training/
  bytes.py
  schema.py
  dataset.py
  model.py
  train.py
  evaluate.py
  validate_dataset.py
  native_brain.py
  experiment6_learned_operations.py
  test_*.py
  README*.md
  colab/
    mirror7_training.ipynb
~~~

## Documentation arrangement

~~~text
docs/
  README.md
  01-project-overview.md
  02-system-architecture.md
  03-backend.md
  04-training.md
  05-evaluation.md
  06-testing-ci.md
  07-repository-organization.md
  08-limitations.md
  09-roadmap.md
  research/
  archive/
~~~

## Why phase folders stay where they are

Flattening hundreds of phase directories would destroy the relationship between implementation, tests, and the research question that created them.

The better arrangement is a strong human-facing map while preserving original source structure.

## Root files

Current entry points such as README.md, pyproject.toml, Docker files, requirements, license, and project metadata remain at the root.

Historical phase reports are retained at their existing paths to avoid breaking cross-references and provenance. The current docs index them without forcing a risky bulk rename.
