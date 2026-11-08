CIFAR-10 CSV deliverables
=========================
train.csv (50,000 rows x 3,073 cols) and test.csv (10,000 rows x 3,073 cols) are
produced exactly as specified in the project's data protocol. Their combined size is
~629 MB, which exceeds typical upload limits, so this folder ships:

  * dataset_manifest.csv          -- row/column counts, byte sizes and SHA-256 digests
  * train_sample_100rows.csv      -- first 100 rows, showing the exact column layout
  * test_sample_100rows.csv       -- ditto for the test split

To regenerate the full files (one command, ~1 minute):

  1. Download https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  2. tar -xzf cifar-10-python.tar.gz -C extracted_files
  3. python code/scripts_build_csv.py

The SHA-256 digests above can be used to verify the regenerated files are identical.
