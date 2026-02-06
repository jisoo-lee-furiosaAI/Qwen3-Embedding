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
  --output_dir results/${model_name} \
  --batch_size 1 \
  --tasks "${tasks}"