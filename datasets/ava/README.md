# Tested with version v2.2 of the AVA dataset

- Download dataset from: https://research.google.com/ava/download.html
- Process videos with `extract_segments_ava.py` with (please do not modify the provided action list csv file): 
```bash
python extract_segments_ava.py \
--video_dir ... \
--annot_file ... \
--output_dir ... \
--filter_on_class 10
```