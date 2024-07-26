export OPENAI_API_KEY=5580ab23f46e477a95ab455f361b5fdb
export OPENAI_AZURE_DEPLOYMENT=1 
# python open-eqa-predict.py --method gpt4o --force --verbose
python evaluate-predictions.py data/predictions/gpt4o-predictions.json --force --verbose
