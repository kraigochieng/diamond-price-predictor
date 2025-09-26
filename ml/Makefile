PORT ?= 5001
MODEL_NAME ?= best_model
MODEL_VERSION ?= 1

.PHONY: all data features train figures clean-data clean-models clean-figures clean-all serve

# Default target
all: data features train figures

# === Pipelines ===
data:
	poetry run python ml/data/make_dataset.py

features: data
	poetry run python ml/features/build_features.py

train:
	poetry run python -m ml.modeling.train_model

figures: data features
	poetry run python ml/visualisation/eda_plots.py

# === Cleaning ===
clean-data:
	rm -rf data/interim/*.csv data/processed/*.csv

clean-models:
	rm -rf models/*.pkl mlruns/

clean-figures:
	rm -rf reports/figures/*.png

clean-all: clean-data clean-models clean-figures

serve:
	poetry run mlflow models serve \
		-m models:/$(MODEL_NAME)/$(MODEL_VERSION) \
		-p $(PORT) \
		--env-manager=local