.PHONY: install run clean-index clean help

install:
	pip install -r requirements.txt

run:
	@if [ -z "$(QUESTION)" ]; then \
		echo "Usage: make run QUESTION='Your question here'"; \
		exit 1; \
	fi
	python openai_agent.py "$(QUESTION)"

clean-index:
	rm -rf data/whoosh_index/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

help:
	@echo "Targets:"
	@echo "  install      - Install dependencies"
	@echo "  run          - Run agent (requires QUESTION and OPENAI_API_KEY)"
	@echo "  clean-index  - Clear cached index"
	@echo "  clean        - Clean temporary files"
