```
mkdir -p \
	data/raw data/processed data/external \
	notebooks \
	src/{data,features,models,utils} \
	app \
	configs \
	scripts \
	tests \
	.github/workflows \
	docker \
	great_expectations \
	mlruns \
	artifacts
```

```commandline
cat > requirements.txt << 'EOF'
pandas
numpy
scikit-learn
mlflow
fastapi
uvicorn
pydantic
python-dotenv
joblib
great-expectations
pytest
EOF

```

```commandline
python3.11 -m venv venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```