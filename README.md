# Fork of CodeT5 for DL course

This is a fork of CodeT5 for fixinig image recognition errors in [ImageToCode](https://github.com/llesha/image-to-code) project

Notebook with learning: [CodeT5/TuningT5.ipynb](https://github.com/MassterMax/CodeT5/blob/main/TuningT5.ipynb)

Tuned model is [here](https://drive.google.com/file/d/19Sb_aMCi-XIBrjqlDiGpYOwG7MmCpWnj/view?usp=sharing)

### Usage:

```python
from transformers import T5Config, RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
config = T5Config.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained("/path/to/bin/model",  config=config)

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
