install-env:
	poetry install

format:
	ruff format src/ 
	isort src/

tests:
	pytest tets/

prepare-docs:
	mkdocs build

local docs: prepare-docs
	mkdocs serve