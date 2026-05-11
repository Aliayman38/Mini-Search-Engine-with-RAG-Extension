"""
Phase 2: Text Preprocessing Pipeline
Steps:
  1. Tokenization
  2. Lowercasing / normalization
  3. Stopword removal
  4. Porter Stemming
Applies the SAME pipeline to both documents and queries.
"""

import re
import string
from cranfield_parser import parse_documents, parse_queries, parse_qrels


# ─────────────────────────────────────────────
# Stopwords (loaded from local NLTK file)
# ─────────────────────────────────────────────
def load_stopwords() -> set:
    """Load English stopwords from NLTK."""
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))


# ─────────────────────────────────────────────
# Porter Stemmer (pure Python, no dependencies)
# ─────────────────────────────────────────────
class PorterStemmer:
    """
    A clean, self-contained implementation of the Porter Stemming Algorithm.
    Reference: M.F. Porter, "An algorithm for suffix stripping", 1980.
    """

    VOWELS = set('aeiou')

    def _is_vowel(self, word, i):
        c = word[i]
        if c in self.VOWELS:
            return True
        if c == 'y' and i > 0 and word[i - 1] not in self.VOWELS:
            return True
        return False

    def _measure(self, stem):
        """Count VC sequences (m) in stem."""
        n = len(stem)
        i, m = 0, 0
        in_vowel_seq = False
        for i in range(n):
            if self._is_vowel(stem, i):
                in_vowel_seq = True
            else:
                if in_vowel_seq:
                    m += 1
                    in_vowel_seq = False
        return m

    def _has_vowel(self, stem):
        return any(self._is_vowel(stem, i) for i in range(len(stem)))

    def _ends_double_consonant(self, word):
        return (len(word) >= 2
                and word[-1] == word[-2]
                and word[-1] not in self.VOWELS)

    def _ends_cvc(self, word):
        if len(word) < 3:
            return False
        c1, v, c2 = word[-3], word[-2], word[-1]
        return (c1 not in self.VOWELS
                and v in self.VOWELS
                and c2 not in self.VOWELS
                and c2 not in 'wxy')

    def _replace_suffix(self, word, suffix, replacement, min_m=0):
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if self._measure(stem) > min_m:
                return stem + replacement
        return word

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word
        word = word.lower()

        # Step 1a
        if word.endswith('sses'):
            word = word[:-2]
        elif word.endswith('ies'):
            word = word[:-2]
        elif word.endswith('ss'):
            pass
        elif word.endswith('s'):
            word = word[:-1]

        # Step 1b
        def step1b_fix(word):
            suffixes = [('eed', 'ee', 0, False), ('ed', '', -1, True), ('ing', '', -1, True)]
            for suf, rep, m_cond, needs_vowel in suffixes:
                if word.endswith(suf):
                    stem = word[:len(word) - len(suf)]
                    cond = (self._measure(stem) > 0) if suf == 'eed' else self._has_vowel(stem)
                    if cond:
                        word = stem + rep
                        if suf != 'eed':
                            # additional rules
                            if word.endswith(('at', 'bl', 'iz')):
                                word += 'e'
                            elif self._ends_double_consonant(word) and not word.endswith(('l', 's', 'z')):
                                word = word[:-1]
                            elif self._measure(word) == 1 and self._ends_cvc(word):
                                word += 'e'
                    break
            return word
        word = step1b_fix(word)

        # Step 1c
        if word.endswith('y') and self._has_vowel(word[:-1]):
            word = word[:-1] + 'i'

        # Step 2
        step2_map = [
            ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'),
            ('izer', 'ize'), ('abli', 'able'), ('alli', 'al'), ('entli', 'ent'),
            ('eli', 'e'), ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
            ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
            ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble'),
        ]
        for suf, rep in step2_map:
            if word.endswith(suf):
                stem = word[:len(word) - len(suf)]
                if self._measure(stem) > 0:
                    word = stem + rep
                break

        # Step 3
        step3_map = [
            ('icate', 'ic'), ('ative', ''), ('alize', 'al'),
            ('iciti', 'ic'), ('ical', 'ic'), ('ful', ''), ('ness', ''),
        ]
        for suf, rep in step3_map:
            if word.endswith(suf):
                stem = word[:len(word) - len(suf)]
                if self._measure(stem) > 0:
                    word = stem + rep
                break

        # Step 4
        step4_suffixes = [
            'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement',
            'ment', 'ent', 'ion', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize',
        ]
        for suf in step4_suffixes:
            if word.endswith(suf):
                stem = word[:len(word) - len(suf)]
                m = self._measure(stem)
                if suf == 'ion':
                    if m > 1 and stem and stem[-1] in 'st':
                        word = stem
                elif m > 1:
                    word = stem
                break

        # Step 5a
        if word.endswith('e'):
            stem = word[:-1]
            m = self._measure(stem)
            if m > 1 or (m == 1 and not self._ends_cvc(stem)):
                word = stem

        # Step 5b
        if (self._measure(word) > 1
                and self._ends_double_consonant(word)
                and word.endswith('l')):
            word = word[:-1]

        return word


# ─────────────────────────────────────────────
# Preprocessing Pipeline
# ─────────────────────────────────────────────
class TextPreprocessor:
    def __init__(self):
        self.stopwords = load_stopwords()
        self.stemmer = PorterStemmer()

    def preprocess(self, text: str) -> list:
        """
        Full preprocessing pipeline:
          1. Lowercase
          2. Remove punctuation & numbers
          3. Tokenize (split on whitespace)
          4. Remove stopwords
          5. Stem with Porter Stemmer
          6. Remove very short tokens (len < 2)

        Returns:
            list of processed tokens
        """
        # 1. Lowercase
        text = text.lower()

        # 2. Remove punctuation and digits
        text = re.sub(r'[^a-z\s]', ' ', text)

        # 3. Tokenize
        tokens = text.split()

        # 4. Stopword removal
        tokens = [t for t in tokens if t not in self.stopwords]

        # 5. Stemming
        tokens = [self.stemmer.stem(t) for t in tokens]

        # 6. Remove very short tokens
        tokens = [t for t in tokens if len(t) >= 2]

        return tokens

    def preprocess_documents(self, documents: dict) -> dict:
        """
        Preprocess all documents.
        Returns: {doc_id -> list of tokens}
        """
        processed = {}
        for doc_id, doc in documents.items():
            processed[doc_id] = self.preprocess(doc['content'])
        return processed

    def preprocess_queries(self, queries: dict) -> dict:
        """
        Preprocess all queries.
        Returns: {query_id -> list of tokens}
        """
        processed = {}
        for qid, text in queries.items():
            processed[qid] = self.preprocess(text)
        return processed


# ─────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────
def verify_preprocessing(documents, queries, proc_docs, proc_queries):
    print("=" * 55)
    print("     PHASE 2: PREPROCESSING VERIFICATION")
    print("=" * 55)

    # Document example
    doc_id = 1
    original_content = documents[doc_id]['content']
    processed_tokens = proc_docs[doc_id]
    print(f"\n--- Document {doc_id} ---")
    print(f"  Original  : {original_content[:120]}...")
    print(f"  Tokens    : {processed_tokens[:15]}...")
    print(f"  Token count: {len(processed_tokens)}")

    # Query example
    q_id = list(queries.keys())[0]
    print(f"\n--- Query {q_id} ---")
    print(f"  Original  : {queries[q_id]}")
    print(f"  Tokens    : {proc_queries[q_id]}")

    # Vocabulary stats
    vocab = set()
    total_tokens = 0
    for tokens in proc_docs.values():
        vocab.update(tokens)
        total_tokens += len(tokens)

    print(f"\n--- Vocabulary Stats ---")
    print(f"  Vocabulary size    : {len(vocab):,} unique terms")
    print(f"  Total tokens (docs): {total_tokens:,}")
    avg = total_tokens / len(proc_docs)
    print(f"  Avg tokens/doc     : {avg:.1f}")

    # Stemmer demo
    stemmer = PorterStemmer()
    test_words = ['aerodynamics', 'efficiency', 'compressible', 'flowing',
                  'stability', 'equations', 'turbulent', 'investigation']
    print(f"\n--- Stemmer Demo ---")
    for w in test_words:
        print(f"  {w:25s} → {stemmer.stem(w)}")

    print(f"\n✅ Phase 2 Complete — preprocessing pipeline ready!")
    print("=" * 55)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    DOCS_PATH    = '/mnt/user-data/uploads/data/cran.all.1400'
    QUERIES_PATH = '/mnt/user-data/uploads/data/cran.qry'
    QRELS_PATH   = '/mnt/user-data/uploads/cran_data/cranqrel'

    print("Loading data...")
    documents = parse_documents(DOCS_PATH)
    queries   = parse_queries(QUERIES_PATH)

    print("Preprocessing...")
    preprocessor = TextPreprocessor()
    proc_docs    = preprocessor.preprocess_documents(documents)
    proc_queries = preprocessor.preprocess_queries(queries)

    verify_preprocessing(documents, queries, proc_docs, proc_queries)
