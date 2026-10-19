# syntax=docker/dockerfile:1
# Sandbox image: runs model-generated code and the official EvalPlus evaluator.
# Containers from this image are started with --network none and CPU / memory /
# process limits (see hmas/sandbox.py).
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# evalplus is installed without its LLM-provider extras (transformers, vllm, ...);
# only the evaluator's own dependencies are needed inside the sandbox.
# The optional "pip_cert" secret is only needed behind a TLS-intercepting proxy.
RUN --mount=type=secret,id=pip_cert,required=false \
    if [ -f /run/secrets/pip_cert ]; then export PIP_CERT=/run/secrets/pip_cert; fi; \
    pip install --no-deps evalplus==0.3.1 \
 && pip install numpy tqdm termcolor fire "tree-sitter>=0.22" "tree-sitter-python>=0.21" \
        psutil appdirs wget multipledispatch tempdir rich datasets pytest

# Backport of an upstream fix for the HumanEval/32 oracle (see the script).
COPY patches/fix_find_zero.py /opt/patches/fix_find_zero.py
RUN python /opt/patches/fix_find_zero.py

RUN useradd -m -u 1000 runner
USER runner
WORKDIR /work
