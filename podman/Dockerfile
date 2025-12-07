# podman build -f podman/Dockerfile -t manylinux2014:latest
FROM ghcr.io/rust-cross/manylinux2014-cross:x86_64

RUN apt-get update && apt-get install -y pkg-config openssl libssl-dev

WORKDIR /build

RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin

# always try to update rust
ARG CACHEBUST=1

RUN sh -c "curl https://sh.rustup.rs -sSf | sh -s -- -y"
