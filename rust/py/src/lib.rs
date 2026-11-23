//! Binding crate: the private `ef_rust._core` extension module.
//!
//! Thin shell over `ef-hashing-core`: reads a Python list of lowercased `str`s and adds
//! their un-normalised hashed counts, in place, into a zeroed `(n, dim)` float32 `NumPy`
//! array the caller allocated. Defaults, validation, allocation, lowercasing and
//! normalisation belong to ef's facade (`ef.embedders.HashingEmbedder(backend="rust")`).
//! The output is allocated by `NumPy`, not here, so its size errors, its allocator and
//! its flags are the Python path's own.
//!
//! Domain routing: an item that is not exactly a `str`, or a `str` that is not valid
//! UTF-8 (a lone surrogate), is left untouched and its index returned in `skipped`; the
//! facade computes those rows with the Python path, so its errors and results stay the
//! reference's.
//!
//! Interpreter and threads: the kernel borrows each `str`'s UTF-8 buffer and the
//! caller's array, so it runs attached; the module declares `gil_used = true`. The
//! `HashingKernel` handle is immutable (`frozen`) and each call allocates its own
//! scratch buffers.

use ef_hashing_core::Error;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The one place core errors become Python exceptions (used as `.map_err(to_pyerr)`).
#[expect(
    clippy::needless_pass_by_value,
    reason = "the shape `.map_err` takes: FnOnce(Error) -> PyErr"
)]
fn to_pyerr(err: Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Native kernel of `ef.embedders.HashingEmbedder`; not part of any public API.
// gil_used = true: the kernel reads borrowed `str` and array buffers while attached.
#[pymodule(gil_used = true)]
mod _core {
    use ef_hashing_core::{Config, HashingKernel, Scratch, WordClasses};
    use numpy::{PyReadwriteArray2, PyUntypedArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyList, PyString};

    use super::to_pyerr;

    /// True in debug builds, so benchmarks can refuse them.
    #[pymodule_export]
    const DEBUG_BUILD: bool = cfg!(debug_assertions);

    /// The hashing kernel for one `HashingEmbedder` configuration.
    #[pyclass(frozen, module = "ef_rust._core", name = "HashingKernel")]
    struct PyHashingKernel {
        inner: HashingKernel,
    }

    #[pymethods]
    impl PyHashingKernel {
        /// Build a kernel; `word_ranges` are the half-open code-point ranges `\w` matches.
        #[new]
        #[pyo3(signature = (word_ranges, /, *, dim, ngram_min, ngram_max, sublinear_tf))]
        fn new(
            word_ranges: Vec<(u32, u32)>,
            dim: usize,
            ngram_min: usize,
            ngram_max: usize,
            sublinear_tf: bool,
        ) -> PyResult<Self> {
            let classes = WordClasses::from_ranges(word_ranges).map_err(to_pyerr)?;
            let config = Config {
                dim,
                ngram_min,
                ngram_max,
                sublinear_tf,
            };
            let inner = HashingKernel::new(classes, config).map_err(to_pyerr)?;
            Ok(Self { inner })
        }

        /// Add the hashed counts of `texts` into the C-contiguous float32 array `out` of
        /// shape `(len(texts), dim)`, row by row; return the indices of the items left
        /// for the Python path (not an exact `str`, or a `str` with a lone surrogate).
        #[pyo3(signature = (texts, out, /))]
        fn accumulate_into(
            &self,
            texts: &Bound<'_, PyList>,
            mut out: PyReadwriteArray2<'_, f32>,
        ) -> PyResult<Vec<usize>> {
            let rows = texts.len();
            let dim = self.inner.dim();
            let shape = out.shape();
            if shape != [rows, dim] {
                return Err(PyValueError::new_err(format!(
                    "out has shape {shape:?}, expected [{rows}, {dim}]"
                )));
            }
            let buffer = out
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("out must be C-contiguous"))?;
            let mut scratch = Scratch::default();
            let mut skipped = Vec::new();
            if dim == 0 || rows == 0 {
                return Ok(skipped);
            }
            for (i, (item, row)) in texts.iter().zip(buffer.chunks_exact_mut(dim)).enumerate() {
                let text = item
                    .cast_exact::<PyString>()
                    .ok()
                    .and_then(|s| s.to_str().ok());
                match text {
                    Some(text) => self
                        .inner
                        .accumulate(text, row, &mut scratch)
                        .map_err(to_pyerr)?,
                    None => skipped.push(i),
                }
            }
            Ok(skipped)
        }

        /// The output width.
        #[getter]
        fn dim(&self) -> usize {
            self.inner.dim()
        }
    }
}
