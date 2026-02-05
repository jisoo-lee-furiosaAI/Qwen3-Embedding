export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS='8'
model_path="BAAI/bge-m3"
model_name="BAAI/bge-m3"

tasks="TwentyNewsgroupsClustering.v2,TweetSentimentExtractionClassification,SCIDOCS,TwitterSemEval2015,STS22.v2,SummEvalSummarization.v2,MindSmallReranking"
# tasks="TwentyNewsgroupsClustering.v2"
# tasks="SCIDOCS,MindSmallReranking"
# tasks="SCIDOCS"

# overall evaluation
python run_mteb.py \
  --model ${model_path} \
  --model_name ${model_name} \
  --backend "openai" \
  --precision bf16 \
  --model_kwargs "{\"max_length\": 8192, \"attn_type\": \"causal\", \"pooler_type\": \"last\", \"do_norm\": true, \"use_instruction\": true, \"instruction_template\": \"Instruct: {}\nQuery:\", \"instruction_dict_path\": \"task_prompts.json\", \"attn_implementation\":\"flash_attention_2\"}" \
  --run_kwargs "{\"save_predictions\": \"true\"}" \
  --output_dir results/${model_name} \
  --batch_size 2 \
  --tasks "${tasks}"
