# Fork of CodeT5 for DL course

Notebook with learning: CodeT5/TuningT5.ipynb

Link for tuned model: https://drive.google.com/file/d/19Sb_aMCi-XIBrjqlDiGpYOwG7MmCpWnj/view?usp=sharing

You can use it with provided code:

```python
from transformers import T5Config, RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
config = T5Config.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained("/path/to/bin",  config=config)
```
