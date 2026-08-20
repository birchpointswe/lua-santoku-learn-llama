STUB := $(CURDIR)/stub.pl
LLAMA_COMMIT := e6b4acfe86af380c4e631973b9caa14337954423

results.mk:
	[ -f llama.cpp-$(LLAMA_COMMIT)-src.tar.gz ] || { echo "missing vendored llama.cpp-$(LLAMA_COMMIT)-src.tar.gz; run 'toku vendor'" >&2; exit 1; }
	if [ -d llama.cpp ]; then rm -rf llama.cpp; fi
	mkdir llama.cpp
	tar xf llama.cpp-$(LLAMA_COMMIT)-src.tar.gz -C llama.cpp
	cd llama.cpp/src/models && \
		for f in *.cpp; do \
			case "$$f" in bert.cpp|llama.cpp) ;; \
				*) perl $(STUB) "$$f" ;; \
			esac; \
		done
	cd llama.cpp/src && \
		for f in llama-chat.cpp llama-grammar.cpp llama-quant.cpp llama-model-saver.cpp; do \
			perl $(STUB) "$$f"; \
		done
	cd llama.cpp && cmake -B build \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=OFF \
		-DLLAMA_BUILD_COMMON=OFF \
		-DLLAMA_BUILD_TOOLS=OFF \
		-DGGML_NATIVE=ON \
		-DGGML_OPENMP=ON \
		-DGGML_BLAS=ON \
		-DGGML_BLAS_VENDOR=OpenBLAS \
		-DGGML_CUDA=OFF \
		-DGGML_METAL=OFF \
		-DGGML_VULKAN=OFF \
		-DGGML_OPENCL=OFF \
		-DGGML_SYCL=OFF \
		-DGGML_RPC=OFF \
		-DGGML_LLAMAFILE=OFF
	cd llama.cpp && cmake --build build -j$$(nproc)
	touch results.mk
