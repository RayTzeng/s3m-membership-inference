BASE_PATH=/mnt/storage2/speech_data/LibriSpeech
OUTPUT_PATH=/mnt/storage2/speech_data/LibriSpeech-Feature
MODEL_PATH=/mnt/ship_groups/public/ssm_pretrain/TERA
OUTPUT_PREFIX=mel20ms
CONFIG=/livingrooms/wiz94156/Privacy-issues-speech-bert/s3prl/s3prl/upstream/baseline/mel20ms.yaml
for MODEL in pretrain_libri_reverse
do
echo "running ${MODEL}"

for SPLIT in train-clean-100 test-clean test-other dev-clean dev-other
do
echo "[running ${SPLIT}]"
OUTPUT_PREFIX=$MODEL
python preprocess_feature_LibriSpeech.py --base_path $BASE_PATH \
    --split $SPLIT \
    --output_path $OUTPUT_PATH \
    --model tera_100hr \
    --state_dict $MODEL_PATH/$MODEL/states-epoch-224.ckpt \
    --model_cfg $MODEL_PATH/$MODEL/config_model.yaml \
    --output_prefix $OUTPUT_PREFIX \
    --config $CONFIG
done

done
