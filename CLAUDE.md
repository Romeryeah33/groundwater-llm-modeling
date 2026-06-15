# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Physical-LLM Groundwater Intelligent Modeling System — a Python application that lets users build groundwater flow/transport MODFLOW 6 models through natural language conversation. A Gradio web UI sits on top of an LLM Agent pipeline that parses intent → builds a JSON IR → validates against physics constraints → invokes FloPy to run MODFLOW 6/MT3DMS → visualizes results → generates a Word report.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize the semantic database (required once)
python scripts/init_semantic_db.py

# Run the main Gradio frontend (http://127.0.0.1:7860)
python -m app.frontend.gradio_app

# Run the FastAPI backend standalone
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Run specific test suites
python tests/test_llm_agent_import.py       # LLM agent modules
python tests/test_physics_engine.py          # Physics constraint validation
python tests/test_model_builder.py           # FloPy auto-modeling E2E
python tests/test_analysis_import.py         # Visualization & report modules

# Run MODFLOW 6 baseline verification (creates a simple 2D confined aquifer)
python scripts/flopy_baseline_test.py

# Docker
docker compose -f docker/docker-compose.yml up
```

## Architecture: the core pipeline

The system processes input through a linear pipeline. Each stage reads/transforms a shared JSON IR (Intermediate Representation) that is the single source of truth for model parameters.

```
User NL / files → IntentParser → DialogueManager → JSON IR
  → ClosedLoopValidator (6-layer physics check) → AutoModelBuilder (FloPy)
  → SolverMonitor → VisualizationEngine → ReportGenerator
```

**JSON IR** (`docs/ir_schema_v1.0.json`) is the contract between all stages. Every module reads from or writes to it. The schema covers 12 model-element categories: aquifer, grid, boundary_conditions, stress_periods, transport, solver, output, etc. Each parameter carries `value`, `unit`, `valid_range`, `source` fields.

**DialogueManager** owns conversation state (`intake → clarification → confirmation → execution`). It delegates parsing to `IntentParser` (5-step pipeline: classify intent → NER → semantic-DB match → generate JSON IR → detect missing params). Missing parameters are classified into three priority levels; priority-1 (blocking) params prevent model execution.

**ClosedLoopValidator** applies six constraint layers (schema → param ranges → unit consistency → physics laws → numerical stability → FloPy pre-run), calling LLMs to re-generate corrected JSON IR for violations. `physics_engine/constraints.py` implements the five hard-coded constraint classes (MassBalance, GoverningEquation, ParameterRange, UnitConsistency, NumericalStability). `quick_validate()` does a no-LLM fast path for real-time frontend feedback.

**AutoModelBuilder** maps JSON IR → MODFLOW 6 via FloPy. `model_mapper.py` contains the IR→FloPy mapping table. `grid_generator.py` produces uniform/refined grids. `solver_monitor.py` classifies MODFLOW errors (non-convergence, dry cells, negative concentration, etc.) and applies adaptive strategies (reduce time step, tighten convergence, increase dispersion).

**VisualizationEngine** produces matplotlib figures (head contours with flow arrows, plume evolution sequences, breakthrough curves, process-contribution pies, dashboards). Set to `Agg` backend for non-interactive use.

## Key design decisions

- **No LLM required for core physics checks.** `ParameterRangeConstraint`, `NumericalStabilityConstraint`, etc. are deterministic Python functions. LLMs are only invoked for NER, intent classification, and correction-prompt regeneration — never for physics enforcement.
- **Semantic DB** (`data/semantic_db.sqlite`) stores 36 term aliases, 16 physical parameter ranges, and 15 unit conversions. `app/knowledge/semantic_db.py` provides fuzzy search, range queries, and value suggestion. Regenerate with `python scripts/init_semantic_db.py` (destroys existing DB).
- **Gradio state management.** Each browser session gets a `DialogueManager` instance stored in a global `sessions` dict keyed by session ID. Reset a session by sending "清空" or "reset".
- **Python 3.11+ required.** Windows encoding quirks: always set `$env:PYTHONIOENCODING='utf-8'` before running scripts, or include the `io.TextIOWrapper` reconfiguration at the top of scripts that print Chinese characters.
- **MODFLOW 6 binary** is auto-downloaded by FloPy on first use. Linux Docker images pre-package it (see `docker/Dockerfile`).

## Module directory map

| Directory | Purpose | Key files |
|-----------|---------|-----------|
| `app/llm_agent/` | NL understanding, dialogue state, file parsing | `intent_parser.py`, `dialogue_manager.py`, `prompt_templates.py`, `file_handler.py`, `llm_client.py` |
| `app/physics_engine/` | Constraint validation | `constraints.py` (5 classes), `validator.py` (loop), `reaction_pathway.py` |
| `app/modeling/` | FloPy model construction | `grid_generator.py`, `model_mapper.py`, `model_builder.py`, `solver_monitor.py` |
| `app/analysis/` | Post-processing | `result_processor.py`, `visualizer.py`, `process_identifier.py`, `report_generator.py` |
| `app/knowledge/` | Semantic DB | `semantic_db.py`, `db_init.sql` |
| `app/frontend/` | UI | `gradio_app.py` |
| `app/api/` | REST endpoints | `routes.py` |
| `app/` root | Entry & config | `main.py` (FastAPI), `config.py` (Settings dataclass) |

## Config

All settings live in `app/config.py` as a `Settings` dataclass. LLM provider/model/API-key are read from environment variables (with `.env` file support). Key env vars: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_API_BASE`, `GRADIO_PORT`.

## Adding a new physics constraint

1. Add a class or `@staticmethod` to `app/physics_engine/constraints.py` returning `ConstraintResult`.
2. Add a `_check_*` method to `ClosedLoopValidator` in `validator.py`.
3. Wire it into the `validate()` loop at the appropriate layer.
4. Add a test case to `tests/test_physics_engine.py`.

## Adding a new file format to FileUploadHandler

1. Add extension → category mapping in `FileUploadHandler.SUPPORTED_EXTENSIONS`.
2. Implement `_parse_<format>()` returning a dict with at minimum `suggested_use`.
3. Register the parser call in `parse_files()`.
