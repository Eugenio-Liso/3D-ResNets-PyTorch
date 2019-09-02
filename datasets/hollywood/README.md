Download dataset here: https://www.di.ens.fr/~laptev/actions/hollywood2/

Run the `extract_segments_hollywood.py` script with:
```bash
python extract_segments_hollywood.py \
--video_dir .../Hollywood2/AVIClips \
--annot_file .../Hollywood2/ClipSets/Run_train.txt \
--output_dir ...

python extract_segments_hollywood.py \
--video_dir .../Hollywood2/AVIClips \
--annot_file .../Hollywood2/ClipSets/Run_test.txt \
--output_dir ...
```