---
library_name: transformers
license: cc-by-nc-sa-4.0
pipeline_tag: feature-extraction
tags:
- protein language model
datasets:
- UniRef50
---

# Paper title and link

The model was presented in the paper [Ankh3: Multi-Task Pretraining with Sequence Denoising and Completion Enhances Protein Representations](https://huggingface.co/papers/2505.20052).

# Paper abstract

The abstract of the paper is the following:

Protein language models (PLMs) have emerged as powerful tools to detect complex patterns of protein sequences. However, the capability of PLMs to fully capture information on protein sequences might be limited by focusing on single pre-training tasks. Although adding data modalities or supervised objectives can improve the performance of PLMs, pre-training often remains focused on denoising corrupted sequences. To push the boundaries of PLMs, our research investigated a multi-task pre-training strategy. We developed Ankh3, a model jointly optimized on two objectives: masked language modeling with multiple masking probabilities and protein sequence completion relying only on protein sequences as input. This multi-task pre-training demonstrated that PLMs can learn richer and more generalizable representations solely from protein sequences. The results demonstrated improved performance in downstream tasks, such as secondary structure prediction, fluorescence, GB1 fitness, and contact prediction. The integration of multiple tasks gave the model a more comprehensive understanding of protein properties, leading to more robust and accurate predictions.

# Model Details:
Ankh3 is a protein language model that is jointly optimized on two objectives:
* Masked language modeling with multiple masking probabilities
* Protein sequence completion.

This is the model of the paper [Ankh3: Multi-Task Pretraining with Sequence Denoising and Completion Enhances Protein Representations](https://huggingface.co/papers/2505.20052).

Code: https://github.com/agemagician/Ankh

1. Masked Language Modeling:
  - The idea of this task is to intentionally 'corrupt' an input protein sequence by
    masking a certain percentage (X%) of its individual tokens (amino acids),
    and then train the model to reconstruct the original sequence.
  
  - Example on a protein sequence before and after corruption:

    Original protein sequence: MKAYVLINSRGP

    This sequence will be masked/corrupted using sentinel tokens as shown below:
    Sequence after corruption: M <extra_id_0> A Y <extra_id_1> L I <extra_id_2> S R G <extra_id_3>


    The decoder learns to correspond each sentinel token to the actual amino acid that was masked.
    In this example: <extra_id_0> K means that <extra_id_0> corresponds to the "K" amino acid and so on.

    Decoder output: <extra_id_0> K <extra_id_1> V <extra_id_2> N <extra_id_3> P



2. Protein Sequence Completion:
- The idea of this task is to cut the input sequence into
  two segments, where the first segment is fed to the encoder
  and the decoder is tasked to auto-regressively generate the
  second segment conditioned on the first segment representation
  outputted from the encoder.

- Example on protein sequence completion:

  Original sequence: MKAYVLINSRGP

  We will pass "MKAYVL" of it to the encoder, and the decoder is trained
  that given the representation of the first part provided by the encoder,
  it should output the second part which is: "INSRGP"



# How to use:

## For Embedding Extraction:
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5EncoderModel
import torch

# Random sequence from uniprot, most likely Ankh3 saw it during pre-training.
sequence = "MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCLGFLALAYSIKARDQKVVGDLEAARRFGSKAKCYNILAAMWTLVPPLLLLGLVVTGALHLARLAKDSAAFFSTKFDDADYD"

ckpt = "ElnaggarLab/ankh3-large"

# Make sure that you must use `T5Tokenizer` not `AutoTokenizer`.
tokenizer = T5Tokenizer.from_pretrained(ckpt)

# To use the encoder representation using the NLU prefix:
encoder_model = T5EncoderModel.from_pretrained(ckpt).eval()


# For extracting embeddings, consider trying the '[S2S]' prefix.
# Since this prefix was specifically used to denote sequence completion
# during the model's pre-training, its use can sometimes
# lead to improved embedding quality.

nlu_sequence = "[NLU]" + sequence
encoded_nlu_sequence = tokenizer(nlu_sequence, add_special_tokens=True, return_tensors="pt", is_split_into_words=False)

with torch.no_grad():
  embedding = encoder_model(**encoded_nlu_sequence)
```

## For Sequence Completion:
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.generation import GenerationConfig
import torch

sequence = "MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCLGFLALAYSIKARDQKVVGDLEAARRFGSKAKCYNILAAMWTLVPPLLLLGLVVTGALHLARLAKDSAAFFSTKFDDADYD"

ckpt = "ElnaggarLab/ankh3-large"
tokenizer = T5Tokenizer.from_pretrained(ckpt)
# To use the sequence to sequence task using the S2S prefix:
model = T5ForConditionalGeneration.from_pretrained(ckpt).eval()


half_length = int(len(sequence) * 0.5)
s2s_sequence = "[S2S]" + sequence[:half_length]
encoded_s2s_sequence = tokenizer(s2s_sequence, add_special_tokens=True, return_tensors="pt", is_split_into_words=False)
# + 1 to account for the start of sequence token.
gen_config = GenerationConfig(min_length=half_length + 1, max_length=half_length + 1, do_sample=False, num_beams=1)
generated_sequence = model.generate(encoded_s2s_sequence["input_ids"], gen_config, )
predicted_sequence = sequence[:half_length] + tokenizer.batch_decode(generated_sequence)[0]
```