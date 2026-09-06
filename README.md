<p align="center">
  <img src="https://santoku.dev/logo-santoku-learn-llama.png" height="64" alt="santoku-learn-llama">
</p>

# santoku-learn-llama

A C binding to [llama.cpp](https://github.com/ggml-org/llama.cpp) with two modes: turn a
batch of texts into dense embedding vectors, or generate text (raw, or through a chat
template). The embeddings come back as a santoku-matrix `fvec` ready to wrap in an `mtx`
and hand to the santoku-learn KRR pipeline, as the dense alternative to sparse ngrams.

## Documentation

Runnable examples and the full API: [santoku.dev](https://santoku.dev/#santoku-learn-llama).

For agents and LLM tooling: [llms.txt](https://santoku.dev/llms.txt) for the index,
[llms-full.txt](https://santoku.dev/llms-full.txt) for every documented example.

## License

MIT, see [LICENSE](LICENSE).

