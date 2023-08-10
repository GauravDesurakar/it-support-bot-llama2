# Startup

1. Create virtual environment python -m venv <env_name>
2. Activate it:
   - Windows:.\<env_name>\Scripts\activate
3. Download LLM Model from below link
   -   https://huggingface.co/TheBloke/Llama-2-7B-GGML/tree/main
   - Model name: llama-2-7b.ggmlv3.q8_0.bin
   - Keep model under 'llm_model' folder
4. Install the required dependencies pip install -r requirements.txt
5. Run ingest.py. This will create index.faiss and index.pkl file into defined folder
6. Run app.py on command prompt.
   - chainlit run app.py -w

# Other Reference
- Chainlit: https://github.com/Chainlit/chainlit
- Langchain: https://github.com/langchain-ai/langchain
- LLM: https://huggingface.co/TheBloke

# Screenshot
![alt text](C:\Users\P1350143\Desktop\PROJECTS\LLM\it-support-bot-llama2\it-support-bot.png)