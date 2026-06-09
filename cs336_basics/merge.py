from collections import Counter

def to_pairs(token):
  return zip(*(token, token[1:]))

def found_at(token, pair, pos):
  return token[pos:pos+2] == pair

def merge(token, pair, count):
  (a, b) = pair 
  new_token, ab, i, updates = [], a+b, 0, Counter()
  while(i < len(token)):
    if found_at(token, pair, i):
      if new_token:
        prev = token[i-1]
        updates[(prev, a)] -= count
        updates[(prev, ab)] += count
      updates[(a, b)] -= count
      
      while(True):  # consecutive pairs
        if found_at(token, pair, i+2):
          new_token.append(ab)
          updates[(a, b)] -= count
          updates[(b, a)] -= count
          updates[(ab, ab)] += count
          i += 2
        else:
          break

      look = i + 2
      if look < len(token):
        nxt = token[look]
        updates[(b, nxt)] -= count
        updates[(ab, nxt)] += count

      new_token.append(ab)
      i += 2
    else:
      new_token.append(token[i])
      i += 1
  
  if updates: # merge found
    return tuple(new_token), updates
  
  return None, None
  
  

