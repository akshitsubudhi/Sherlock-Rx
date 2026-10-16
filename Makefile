PY ?= .venv/bin/python
BIN = .venv/bin

.PHONY: setup data train test lint typecheck eval bench api ui chat docker clean

setup:            ## create venv, install, generate data, train NLU
	uv venv .venv --python 3.12 && uv pip install --python $(PY) -e ".[llm,ui,dev]"
	$(MAKE) data train

data:             ## synthetic warehouse with planted root causes (scale 1 ≈ 0.3M rows)
	$(BIN)/sherlock generate --scale 1 --seed 7

train:            ## train tagger + intent classifier, write reports/nlu_report.md
	$(BIN)/sherlock train

test:             ## unit tests + acceptance gates (RCA, NLU, dialogue)
	$(BIN)/pytest -o addopts="" -q

lint:
	$(BIN)/ruff check src tests

typecheck:
	$(BIN)/mypy

eval:             ## all evaluations -> reports/EVAL.md
	$(BIN)/sherlock eval all

bench:            ## scale benchmark -> reports/BENCH.md
	$(BIN)/sherlock bench --scales 1,10,50,100

api:
	$(BIN)/uvicorn sherlock_rx.api.main:app --port 8000 --workers 4

ui:
	$(BIN)/streamlit run src/sherlock_rx/ui/app.py

chat:
	$(BIN)/sherlock chat --debug

docker:
	docker compose up --build

clean:
	rm -rf data models/*.pkl .pytest_cache .mypy_cache .ruff_cache
