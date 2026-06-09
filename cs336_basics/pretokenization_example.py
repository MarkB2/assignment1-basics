import os
import regex as re
import psutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def get_iter(text: str | list[str], pat: re.Pattern[str]):
  if isinstance(text, list):
    for line in text:
      yield from get_iter(line, pat)
  else:
    for match in pat.finditer(text):
      yield match.group()

def split_pattern(special_tokens):
    return re.compile("|".join(re.escape(tok) for tok in special_tokens))

def pre_string_reader(str: str, split_pat: list[str] | re.Pattern):
    # if isinstance(pat, list):
    split_pat = split_pattern(split_pat)
    return re.split(split_pat, str)

def chunk_proc(args):
   file, start, end, pat, split_pat = args
   with open(file, 'rb') as f:
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return Counter(get_iter(re.split(split_pat, chunk), pat))

def pre_file_reader(file: str, pat: re.Pattern = PAT, special_tokens: list[str] = ["<|endoftext|>"]):
    with open(file, "rb") as f:
        num_processes = psutil.cpu_count(False)
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    split_pat = split_pattern(special_tokens)
    chunks = [(file, start, end, pat, split_pat)
        for start, end in zip(boundaries[:-1], boundaries[1:])]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
            pre_toks = list(executor.map(chunk_proc, chunks))

    pre_tokens = Counter()
    for pre_tok in pre_toks:
       pre_tokens += pre_tok

    return pre_tokens

