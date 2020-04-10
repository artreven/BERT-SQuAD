# BERT-SQuAD 

## Training 
Training scripts and pretrained model are from [huggingface](https://github.com/huggingface/pytorch-transformers)

## Run

`run_squad.py`: Fine-tuning on SQuAD for question-answering
This example code fine-tunes BERT on the SQuAD dataset using distributed training on 8 V100 GPUs and Bert Whole Word Masking uncased model to reach a F1 > 93 on SQuAD:

```bash
export SQUAD_DIR='/home/revenkoa/local_data/datasets/QA/SQuAD/1.1'
export OUTPUT_DIR='/home/revenkoa/local_data/BERT_DATA/BERT_SQuAD11'

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR
```
Training with these hyper-parameters gave us the following results:

```bash
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ../models/wwm_uncased_finetuned_squad/predictions.json
{"exact_match": 86.91579943235573, "f1": 93.1532499015869}
```

This is the model provided as bert-large-uncased-whole-word-masking-finetuned-squad.

