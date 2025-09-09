.PHONY: all data features train figures clean-data clean-models clean-figures clean-all

# Default target
all: data features train figures

# === Pipelines ===
data:
	poetry run python ml/data/make_dataset.py

features: data
	poetry run python ml/features/build_features.py

train: features
	poetry run python ml/modeling/train_model.py

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
