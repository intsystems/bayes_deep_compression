install-env:
	poetry install

format:
	ruff format src/ tests/ examples/

tests:
	pytest tets/