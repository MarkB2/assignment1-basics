# %%
from cs336_basics.trainer import train_and_save
from cs336_basics.pretokenizer import Pretokenizer
from collections import Counter

train_and_save('data/owt_valid.txt', 'data/owt_common_valid_vocab', num_workers=4)

# iterator = Pretokenizer().iter_file('data/owt_train.txt', num_workers=4)
# print(len(Counter(iterator)))
