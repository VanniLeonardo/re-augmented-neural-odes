# CPU image used to VERIFY the reproduction pipeline runs on a fresh machine with
# no GPU and no Weights & Biases account. This is the ReScience "clone-and-run"
# acceptance environment: `docker build` then `docker run` executes `make smoke`.
#
# Notes:
# - torch/torchvision are the CPU builds (from the pytorch CPU index) so the image
#   is small and portable; GPU experiments use the conda env / requirements.txt.
# - wandb is deliberately NOT installed: logging defaults to the CSV backend, which
#   proves the code never needs an external account.
# - The only network access smoke needs at runtime is the MNIST dataset download.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_LOGGER=csv \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) CPU torch/torchvision from the pytorch CPU index (pinned).
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1
# 2) The rest of the pinned stack from PyPI (no wandb).
RUN pip install \
        torchdiffeq==0.2.5 \
        numpy==2.4.3 \
        scipy==1.17.1 \
        scikit-learn==1.8.0 \
        matplotlib==3.10.9 \
        pandas==3.0.3 \
        rich==15.0.0 \
        pytest==9.0.3

COPY . .

CMD ["make", "smoke"]
