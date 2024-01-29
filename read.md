# AUTOGPT and autonomous agents in langchain

This part is about writing agents that can run plan for an unspecified task, plan how to execute this and run code and tools to achieve a goal. 

### requirements
make sure you have tesseract installed
```
sudo apt install tesseract-ocr
```
openapi key, dont forget to put it in .env file

### general install instructions

To install the libraries
```
poetry install
```

To run streamlit apps
```
poetry run streamlit <filename>
```


teaching demo: create a MKRL agent that can run a literature search against ArXiv and look up the papers, extract the key information and store it in a db.

### First Exercise: Finetuning a multimodal deep learning model
The code in finetuning.ipynb shows how to take a pretrained multimodel document query model and finetune it on data 
to achieve better results. 

To run the notebook:
```
poetry run jupyter server
```

#### Second Exercise: Guided
download and explore AutoGPTbuild a small tool that can do literature research automatically, extracts key information and puts it in a database, by using autogpt.
autogpt can be found at https://github.com/Significant-Gravitas/AutoGPT

#### Third exercise: Guided
Goal: rebuild a small tool that can do literature research automatically, extracts key information and puts it in a database using langchain

Example query be able to run
```
Look up the latest 5 papers on Arxiv about sentiments analysis with generative AI. 
Extract the key information and a summary and put it in a database called research.db (sqlite) and the table is literature
```

#### Fourth exercise: Onl first part Guided
Create a tool that can parse cost declarations automatically, no matter the input form. This can be an image of a restaurant ticket, and excel file with data. See some examples in the folder example data
- Store the data systematically and try to take into account who is the user uploading the file
- Make a policy about costs, so that for instance a manager can have X costs and an employee only can have Y cost
- Make a report that is automatically sent to a manager, also give it a summary of the costs and if he needs to take an action.
- Try to use a small language model (and if time train one.)