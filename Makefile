PY ?= python

.PHONY: setup models run evaluate benchmark verify-math test dataset calibrate clean

setup:            ## install dependencies
	$(PY) -m pip install -r requirements.txt

models:           ## pre-download all models (~1.4 GB)
	$(PY) scripts/download_models.py

run:              ## run the pipeline over the sample corpus
	$(PY) run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports

evaluate:         ## metrics, baselines, ablations -> evaluation/
	$(PY) evaluate.py

benchmark:        ## out-of-sample paraphrase benchmark on MRPC + PAWS
	$(PY) scripts/benchmark_public_paraphrase.py

verify-math:      ## check the proofs in docs/MATHEMATICS.md on the real embeddings
	$(PY) scripts/verify_math.py

test:
	$(PY) -m pytest -q tests

dataset:          ## regenerate sample_submissions (needs macOS `say` or espeak-ng)
	$(PY) scripts/build_dataset.py

calibrate:        ## refit the AI-likelihood combiner on HC3 + SAGE
	$(PY) scripts/calibrate_ai_detector.py

clean:
	rm -rf reports evaluation/ablation_tmp .pytest_cache
