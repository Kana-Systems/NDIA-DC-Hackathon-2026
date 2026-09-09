# Pretrained model attribution

## Legal-BERT

`nlpaueb/legal-bert-base-uncased`, by the Athens University of
Economics and Business NLP group. Model card:
https://huggingface.co/nlpaueb/legal-bert-base-uncased

Chalkidis, Fergadiotis, Malakasiotis, Aletras and Androutsopoulos,
LEGAL-BERT: The Muppets straight out of Law School, EMNLP Findings 2020.
https://aclanthology.org/2020.findings-emnlp.261/

The model card licenses these weights under CC BY-SA 4.0:
https://creativecommons.org/licenses/by-sa/4.0/
Fine-tuned derivatives retain this attribution and share-alike license, separately
from the application's Apache-2.0 code. CUAD attribution also accompanies them.
Our modification is supervised multi-label fine-tuning on CUAD clause windows.
This is not a model trained to decide FAR/DFARS applicability or legal compliance.

## Additional experimental encoders

`FacebookAI/roberta-large` is distributed under MIT, per its model card:
https://huggingface.co/FacebookAI/roberta-large . Preserve the upstream license
with redistributed artifacts. This general-English encoder is an experimental
comparison, not a federal-law model. CUAD-derived training data remains CC BY 4.0.

`microsoft/deberta-v2-xlarge` is distributed under MIT, per its model card:
https://huggingface.co/microsoft/deberta-v2-xlarge

`microsoft/deberta-v3-large` is distributed under MIT, per its model card:
https://huggingface.co/microsoft/deberta-v3-large

`ai-law-society-lab/CaseLawModernBERT-large` is distributed under Apache 2.0,
per its model card:
https://huggingface.co/ai-law-society-lab/CaseLawModernBERT-large

## Experimental Llama classifier

`meta-llama/Llama-3.1-8B` is distributed under the custom Llama 3.1 Community
License, not Apache 2.0:
https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE

Its Acceptable Use Policy is incorporated into that license:
https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md

Meta announced availability for U.S. government agencies and private-sector
partners supporting defense and national-security work:
https://about.fb.com/news/2024/11/open-source-ai-america-global-security/

The experiment uses only public CUAD data. Do not place ITAR-controlled,
classified, government-sensitive, or private contract data on Runpod. Any
redistributed derivative must retain the packaged `NOTICE` and use a model name
beginning with `Llama`.

