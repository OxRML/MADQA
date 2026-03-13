.PHONY: install-aws install-requirements install-poppler run all

all: run

install-poppler:
	@echo "Installing poppler-utils (PDF processing)..."
	@if command -v apt-get > /dev/null; then \
		sudo apt-get update && sudo apt-get install -y poppler-utils; \
	elif command -v yum > /dev/null; then \
		sudo yum install -y poppler-utils; \
	elif command -v brew > /dev/null; then \
		brew install poppler; \
	else \
		echo "ERROR: Could not detect package manager. Please install poppler-utils manually."; \
		exit 1; \
	fi
	@echo "✓ Poppler installed successfully!"

install-requirements: install-poppler
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✓ Python dependencies installed successfully!"

run: install-requirements
	@echo "Running textract OCR..."
	@if [ -z "$$AWS_ACCESS_KEY_ID" ] || [ -z "$$AWS_SECRET_ACCESS_KEY" ]; then \
		echo ""; \
		echo "Please provide AWS credentials:"; \
		read -p "AWS Access Key ID: " AWS_KEY; \
		read -sp "AWS Secret Access Key: " AWS_SECRET; \
		echo ""; \
		python textract.py -o ocr_output.jsonl --aws-access-key-id $$AWS_KEY --aws-secret-access-key $$AWS_SECRET; \
	else \
		python textract.py -o ocr_output.jsonl --aws-access-key-id $$AWS_ACCESS_KEY_ID --aws-secret-access-key $$AWS_SECRET_ACCESS_KEY; \
	fi
