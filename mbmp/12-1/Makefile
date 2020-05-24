.PHONY: proto
proto:
	protoc --python_out=./src/ --mypy_out=./src/ ./proto/*.proto

.PHONY: mypy
mypy:
	mypy src

.PHONY: solve
solve:
	poetry run python -m src.main
