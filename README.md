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

### Example:

```python
text = """
com.qooqle.protobuf.MessaqeOrBuilder 
/**
*<code>bool is_invalid = 1;</code>
* @return The isInvalid.
*/
boolean getIsInvalid();
"""

input_ids = tokenizer(text, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=len(input_ids[0]))
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```
This prints following:
```
}
com.google.protobuf.MessageOrBuilder 
/**
* <code>bool is_invalid = 1;</code>
* @return The isInvalid.
*/
boolean getIsInvalid();
```
