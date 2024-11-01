install-env:
	poetry install

format:
	ruff format src/ 
	isort src/

tests:
	pytest tets/