# Tested with version v2.2 of the AVA dataset

- Download dataset from: https://research.google.com/ava/download.html
- Create conda env (or something else) with openCV
```bash
conda install opencv3=3.2.0 -c menpo
```
- Process videos with `extract_dataset.py` with (please do not modify the provided action list csv file): 
```bash
python extract_dataset.py \
--video_dir ... \
--annot_file ... \
--output_dir ...
```