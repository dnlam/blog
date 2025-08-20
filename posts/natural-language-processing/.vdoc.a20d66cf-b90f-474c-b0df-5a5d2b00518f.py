# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
#| output: false
#| eval: true
#| warning: false
#| message: false
from fastai.text.all import *
path = untar_data(URLs.IMDB)
# path.ls()
#
#
#
#| echo: true
#| output: true
#| eval: true
#| warning: false
#| message: false
files = get_text_files(path,folders=['train','test','unsup'])
#
#
#
#
#| echo: true
#| output: true
#| eval: true
#| warning: false
#| message: false
txt = files[0].open().read()
txt[:60]
spacy = WordTokenizer()
toks = first(spacy([txt]))

print(coll_repr(toks,30))
#
#
#
#
#
#
#
#
#
#| echo: true
#| output: true
#| eval: true
#| warning: false
#| message: false
txts = L(o.open().read() for o in files[:2000])
def subword(sz):
    sp = SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])
#
#
#
#
#
#| echo: true
#| output: true
#| eval: true
#| warning: false
#| message: false
subword(10000)
#
#
#
#
#
#
#
#
#| echo: true
#| output: true
#| eval: true
#| warning: false
#| message: false
tkn = Tokenizer(spacy)
toks300 = txts[:300].map(tkn)
toks300[0]
num = Numericalize()
num.setup(toks300)
coll_repr(num.vocab,20)
#
#
#
#
#
#
nums = num(toks)[:20]
nums
#
#
#
#
' '.join(num.vocab[o] for o in nums)
#
#
#
#
#
#
#
#
#
#
#
#
#
nums300 = toks300.map(num)
dl = LMDataLoader(nums300)
x,y = first(dl)
x.shape, y.shape
#
#
#
#
#
#
#
#

get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])

dls_lm = DataBlock(
blocks=TextBlock.from_folder(path, is_lm=True),
get_items=get_imdb, splitter=RandomSplitter(0.1)).dataloaders(path, path=path, bs=128, seq_len=80)
#
#
#
#
#
#
#
learn = language_model_learner(
dls_lm, AWD_LSTM, drop_mult=0.3,
metrics=[accuracy, Perplexity()]).to_fp16()
#
#
#
#
#
#| echo: true
#| output: true
#| eval: true
#| cache: true
#| warning: false
#| message: false
learn.fit_one_cycle(1,2e-2)
#
#
#
#
#
#
#| cache: true
# Option 1: Save with FastAI
# learn.save('one_epoch_training')

# Option 2: Save with PyTorch
import torch
model_save_path = learn.path/'models'/'one_epoch_training_torch.pth'
torch.save(learn.model.state_dict(), model_save_path)
# print(f"Model saved to: {model_save_path}")
#
#
#
#
#
#
#
#
#
#| cache: true
# Option 1: Use FastAI's load method
# learn.load('one_epoch_training', strict=False)

# Option 2: Use PyTorch to load the saved model
import torch
model_load_path = learn.path/'models'/'one_epoch_training_torch.pth'
state_dict = torch.load(model_load_path, weights_only=False)
learn.model.load_state_dict(state_dict, strict=False)
# print(f"Model loaded from: {model_load_path}")
#
#
#
#
#
#| cache: true
#| warning: false
#| message: false
#| eval: true
learn.unfreeze()

learn.fit_one_cycle(10,2e-3)
```
#
#
#
#
#| cache: true
learn.save_encoder('finetuned')
```
#
#
#
#
#
#| echo: true
#| output: true
#| eval: true  
#| warning: false
#| message: false

TEXT = "I liked this movie so"

N_WORDS = 40

N_SENTENCES = 2

preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)]
#
#
#
#
print("\n".join(preds))
#
#
#
#
#
#
#
#
#| cache: true
dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| cache: true
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5,
                                metrics=accuracy).to_fp16()
#
#
#
#| cache: true
learn.load_encoder('finetuned')
#
#
#
#
#
#| cache: true
#| warning: false
#| message: false
#| eval: true
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
#
#
#
#
#| cache: true
#| warning: false
#| message: false
#| eval: true
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
#
#
#
#
#| cache: true
#| warning: false
#| message: false
#| eval: true
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
