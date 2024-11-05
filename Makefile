FOLDERS = src/ examples/ tests/

install-env:
	poetry install

format:
	ruff format $(FOLDERS) 
	isort $(FOLDERS)

tests:
	pytest tets/

prepare-docs:
	mkdocs build

local docs: prepare-docs
	mkdocs serve