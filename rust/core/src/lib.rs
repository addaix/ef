//! The feature-hashing kernel of `ef`'s `HashingEmbedder`, with no Python types.
//!
//! [`HashingKernel::accumulate`] reproduces, bit for bit, the counting and accumulation
//! part of the Python reference (`ef/_reference.py`, algorithm version 1):
//!
//! 1. split the (already lowercased) text into maximal runs of "word" characters;
//! 2. for each n-gram size `n` from `ngram_min` to `ngram_max`, take every window of `n`
//!    consecutive words joined by one space;
//! 3. count each distinct token, in first-occurrence order (Python's `Counter` order);
//! 4. hash each token with an 8-byte `BLAKE2b` digest read big-endian: bucket
//!    `(h >> 1) % dim`, sign from the low bit;
//! 5. add `sign * weight` to the bucket, where `weight` is `1 + ln(count)` (sublinear tf)
//!    or `count`, computed in `f64` and rounded to `f32` before an `f32` addition, in the
//!    token order of step 3 (`NumPy`'s NEP 50 rule for `float32 += float`).
//!
//! Lowercasing and L2 normalisation stay in Python, so the two backends share their
//! exact semantics (`str.lower`, BLAS `sdot`). What counts as a word character is not
//! hard-coded either: the caller passes the code-point ranges Python's `re` `\w` matches
//! ([`WordClasses::from_ranges`]), so the tokenizer agrees with the running interpreter's
//! Unicode version by construction.

use std::cmp::Ordering;
use std::hash::BuildHasher;

use blake2b_simd::Params;
use hashbrown::HashTable;
use rustc_hash::FxBuildHasher;

/// One past the largest Unicode scalar value.
const CODE_POINT_LIMIT: u32 = 0x11_0000;

/// The `BLAKE2b` digest length the algorithm uses, in bytes.
const DIGEST_BYTES: usize = 8;

/// What can go wrong when building or calling a [`HashingKernel`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The word-character ranges are not sorted, disjoint, non-empty half-open ranges
    /// of code points.
    #[error(
        "word-character range {index} is ({start}, {end}): ranges must be non-empty, \
         sorted, disjoint and below 0x110000"
    )]
    Ranges {
        /// Position of the offending range.
        index: usize,
        /// Its start.
        start: u32,
        /// Its (exclusive) end.
        end: u32,
    },
    /// The configuration is outside the algorithm's domain.
    #[error("invalid configuration: {0}")]
    Config(&'static str),
    /// The output row does not have `dim` cells.
    #[error("the output row has {got} cells, the kernel's dim is {dim}")]
    RowLength {
        /// The row's length.
        got: usize,
        /// The kernel's `dim`.
        dim: usize,
    },
}

/// The set of "word" characters, as sorted half-open code-point ranges.
#[derive(Debug, Clone)]
pub struct WordClasses {
    ascii: [bool; 128],
    ranges: Vec<(u32, u32)>,
}

impl WordClasses {
    /// Build the set from sorted, disjoint, non-empty half-open ranges `[start, end)`.
    ///
    /// # Errors
    ///
    /// [`Error::Ranges`] when a range is empty, out of order, overlapping, or beyond the
    /// last code point.
    pub fn from_ranges(ranges: Vec<(u32, u32)>) -> Result<Self, Error> {
        let mut previous_end = 0;
        for (index, &(start, end)) in ranges.iter().enumerate() {
            if start >= end || start < previous_end || end > CODE_POINT_LIMIT {
                return Err(Error::Ranges { index, start, end });
            }
            previous_end = end;
        }
        let mut ascii = [false; 128];
        for (code, cell) in (0_u32..).zip(ascii.iter_mut()) {
            *cell = contains(&ranges, code);
        }
        Ok(Self { ascii, ranges })
    }

    /// Whether `c` is a word character.
    pub fn is_word(&self, c: char) -> bool {
        let code = u32::from(c);
        match usize::try_from(code).ok().and_then(|i| self.ascii.get(i)) {
            Some(&is_word) => is_word,
            None => contains(&self.ranges, code),
        }
    }
}

/// Whether `code` falls in one of the sorted, disjoint half-open `ranges`.
fn contains(ranges: &[(u32, u32)], code: u32) -> bool {
    ranges
        .binary_search_by(|&(start, end)| {
            if end <= code {
                Ordering::Less
            } else if start > code {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

/// The vector-affecting parameters of a `HashingEmbedder`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Output width (number of buckets), at least 1.
    pub dim: usize,
    /// Smallest word n-gram size, at least 1.
    pub ngram_min: usize,
    /// Largest word n-gram size, at least `ngram_min`.
    pub ngram_max: usize,
    /// Weight a token by `1 + ln(count)` instead of `count`.
    pub sublinear_tf: bool,
}

/// One distinct token of a text: where its bytes sit in the arena, its digest, its count.
#[derive(Debug, Clone, Copy)]
struct Entry {
    start: usize,
    end: usize,
    digest: u64,
    count: u64,
}

/// Reusable buffers for [`HashingKernel::accumulate`]; one per thread, reused across
/// texts. Distinct tokens are stored once in a byte arena and found through a hash
/// table of entry indices, so counting allocates nothing per token.
#[derive(Debug, Default)]
pub struct Scratch {
    words: Vec<(usize, usize)>,
    table: HashTable<usize>,
    entries: Vec<Entry>,
    arena: Vec<u8>,
    key: String,
}

/// The bytes of `entry` in `arena` (empty if the range were ever out of bounds).
fn entry_bytes<'a>(arena: &'a [u8], entry: Option<&Entry>) -> &'a [u8] {
    entry
        .and_then(|e| arena.get(e.start..e.end))
        .unwrap_or_default()
}

/// The feature-hashing kernel: a word-character set plus a [`Config`].
#[derive(Debug, Clone)]
pub struct HashingKernel {
    classes: WordClasses,
    config: Config,
    params: Params,
}

impl HashingKernel {
    /// A kernel for `config`, splitting words with `classes`.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when `dim` is 0 or the n-gram range is not `1 <= min <= max`.
    pub fn new(classes: WordClasses, config: Config) -> Result<Self, Error> {
        if config.dim == 0 {
            return Err(Error::Config("dim must be at least 1"));
        }
        if config.ngram_min == 0 || config.ngram_max < config.ngram_min {
            return Err(Error::Config("ngram range must satisfy 1 <= min <= max"));
        }
        let mut params = Params::new();
        params.hash_length(DIGEST_BYTES);
        Ok(Self {
            classes,
            config,
            params,
        })
    }

    /// The output width.
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// The `(bucket, sign)` of one token, as `ef`'s `_hash_bucket` computes it.
    pub fn bucket(&self, token: &str) -> (usize, bool) {
        let h = self.digest(token.as_bytes());
        (self.bucket_of(h), h & 1 == 1)
    }

    /// Add the signed, weighted token counts of `text` (already lowercased) into `row`.
    ///
    /// `row` is not cleared and not normalised: the caller passes a zero row and
    /// normalises it afterwards, as the reference does.
    ///
    /// # Errors
    ///
    /// [`Error::RowLength`] when `row` does not have `dim` cells.
    pub fn accumulate(
        &self,
        text: &str,
        row: &mut [f32],
        scratch: &mut Scratch,
    ) -> Result<(), Error> {
        let dim = self.config.dim;
        if row.len() != dim {
            return Err(Error::RowLength {
                got: row.len(),
                dim,
            });
        }
        let Scratch {
            words,
            table,
            entries,
            arena,
            key,
        } = scratch;
        words.clear();
        table.clear();
        entries.clear();
        arena.clear();
        let fx = FxBuildHasher;

        self.split_words(text, words);
        let ngram_max = self.config.ngram_max.min(words.len());
        for n in self.config.ngram_min..=ngram_max {
            for window in words.windows(n) {
                key.clear();
                for (k, &(start, end)) in window.iter().enumerate() {
                    if k > 0 {
                        key.push(' ');
                    }
                    key.push_str(text.get(start..end).unwrap_or_default());
                }
                let bytes = key.as_bytes();
                let hash = fx.hash_one(bytes);
                let found = table
                    .find(hash, |&i| entry_bytes(arena, entries.get(i)) == bytes)
                    .copied();
                if let Some(entry) = found.and_then(|i| entries.get_mut(i)) {
                    entry.count += 1;
                } else {
                    let start = arena.len();
                    arena.extend_from_slice(bytes);
                    entries.push(Entry {
                        start,
                        end: arena.len(),
                        digest: self.digest(bytes),
                        count: 1,
                    });
                    table.insert_unique(hash, entries.len() - 1, |&i| {
                        fx.hash_one(entry_bytes(arena, entries.get(i)))
                    });
                }
            }
        }

        for &Entry {
            digest: h, count, ..
        } in entries.iter()
        {
            #[expect(
                clippy::cast_precision_loss,
                reason = "Python's float(count) rounds the same way"
            )]
            let count = count as f64;
            let weight = if self.config.sublinear_tf {
                1.0 + count.ln()
            } else {
                count
            };
            let signed = if h & 1 == 1 { weight } else { -weight };
            #[expect(
                clippy::cast_possible_truncation,
                reason = "NEP 50: a Python float added to a float32 is rounded to float32 first"
            )]
            let addend = signed as f32;
            if let Some(cell) = row.get_mut(self.bucket_of(h)) {
                *cell += addend;
            }
        }
        Ok(())
    }

    /// Push the byte ranges of the maximal runs of word characters of `text`.
    fn split_words(&self, text: &str, words: &mut Vec<(usize, usize)>) {
        let mut start = None;
        for (i, c) in text.char_indices() {
            if self.classes.is_word(c) {
                if start.is_none() {
                    start = Some(i);
                }
            } else if let Some(s) = start.take() {
                words.push((s, i));
            }
        }
        if let Some(s) = start {
            words.push((s, text.len()));
        }
    }

    /// The 8-byte `BLAKE2b` digest of `bytes`, read big-endian.
    fn digest(&self, bytes: &[u8]) -> u64 {
        let hash = self.params.hash(bytes);
        let mut out = [0_u8; DIGEST_BYTES];
        for (o, b) in out.iter_mut().zip(hash.as_bytes()) {
            *o = *b;
        }
        u64::from_be_bytes(out)
    }

    /// `(h >> 1) % dim`, which is below `dim` and so fits a `usize`.
    fn bucket_of(&self, h: u64) -> usize {
        // dim <= usize::MAX <= u64::MAX on every supported target
        let dim = u64::try_from(self.config.dim).unwrap_or(u64::MAX);
        usize::try_from((h >> 1) % dim).unwrap_or(usize::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `\w` on ASCII only: enough for unit tests (the Python side passes the full table).
    fn ascii_classes() -> WordClasses {
        WordClasses::from_ranges(vec![(48, 58), (65, 91), (95, 96), (97, 123)]).unwrap()
    }

    fn kernel(dim: usize, ngram: (usize, usize), sublinear_tf: bool) -> HashingKernel {
        let config = Config {
            dim,
            ngram_min: ngram.0,
            ngram_max: ngram.1,
            sublinear_tf,
        };
        HashingKernel::new(ascii_classes(), config).unwrap()
    }

    #[test]
    fn buckets_match_python() {
        // ef._reference.hash_bucket("hello", 512) == (62, 1.0); ("hello world", 512) == (168, -1.0)
        let k = kernel(512, (1, 2), true);
        assert_eq!(k.bucket("hello"), (62, true));
        assert_eq!(k.bucket("hello world"), (168, false));
    }

    #[test]
    fn counts_and_weights() {
        let k = kernel(512, (1, 1), false);
        let mut row = vec![0.0_f32; 512];
        k.accumulate("hello hello", &mut row, &mut Scratch::default())
            .unwrap();
        let (bucket, positive) = k.bucket("hello");
        assert_eq!(row[bucket], if positive { 2.0 } else { -2.0 });
        assert_eq!(row.iter().filter(|v| **v != 0.0).count(), 1);
    }

    #[test]
    fn sublinear_weight() {
        let k = kernel(512, (1, 1), true);
        let mut row = vec![0.0_f32; 512];
        k.accumulate("a a a", &mut row, &mut Scratch::default())
            .unwrap();
        let (bucket, positive) = k.bucket("a");
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the kernel's own f64 -> f32 rounding"
        )]
        let w = (1.0 + 3.0_f64.ln()) as f32;
        assert_eq!(row[bucket], if positive { w } else { -w });
    }

    #[test]
    fn ngrams_join_with_one_space() {
        let k = kernel(1 << 20, (2, 2), false);
        let mut row = vec![0.0_f32; 1 << 20];
        k.accumulate("one,,two", &mut row, &mut Scratch::default())
            .unwrap();
        let (bucket, _) = k.bucket("one two");
        assert_eq!(row[bucket].abs(), 1.0);
    }

    #[test]
    fn empty_and_wordless_texts_leave_zeros() {
        let k = kernel(16, (1, 3), true);
        let mut scratch = Scratch::default();
        for text in ["", "  ,;!", "\u{e9}\u{e9}"] {
            let mut row = vec![0.0_f32; 16];
            k.accumulate(text, &mut row, &mut scratch).unwrap();
            assert!(row.iter().all(|v| *v == 0.0), "{text:?}");
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert!(matches!(
            WordClasses::from_ranges(vec![(5, 5)]),
            Err(Error::Ranges { index: 0, .. })
        ));
        assert!(matches!(
            WordClasses::from_ranges(vec![(0, 10), (5, 20)]),
            Err(Error::Ranges { index: 1, .. })
        ));
        assert!(WordClasses::from_ranges(vec![(0, CODE_POINT_LIMIT + 1)]).is_err());
        let bad = Config {
            dim: 0,
            ngram_min: 1,
            ngram_max: 1,
            sublinear_tf: true,
        };
        assert!(HashingKernel::new(ascii_classes(), bad).is_err());
        let k = kernel(8, (1, 1), true);
        let mut row = vec![0.0_f32; 7];
        assert_eq!(
            k.accumulate("x", &mut row, &mut Scratch::default()),
            Err(Error::RowLength { got: 7, dim: 8 })
        );
    }

    #[test]
    fn non_ascii_word_characters_use_the_ranges() {
        let classes = WordClasses::from_ranges(vec![(97, 123), (0xe9, 0xea)]).unwrap();
        assert!(classes.is_word('\u{e9}'));
        assert!(!classes.is_word('\u{e8}'));
        assert!(classes.is_word('z'));
        assert!(!classes.is_word('Z'));
    }
}
