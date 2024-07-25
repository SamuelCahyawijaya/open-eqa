export OPENAI_API_KEY=5580ab23f46e477a95ab455f361b5fdb 
export OPENAI_AZURE_DEPLOYMENT=1 
python open-eqa-predict.py --method blind-gpt4 --force --verbose
python evaluate-predictions.py data/predictions/blind-gpt4-predictions.json --force --verbose
