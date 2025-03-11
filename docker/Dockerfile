FROM rust:1.85 as build

RUN apt-get update && apt-get install -y clang

WORKDIR /awdit
COPY Cargo.toml Cargo.lock build.rs ./
COPY src src/
COPY tools/dbcop tools/dbcop/

# Build AWDIT
RUN cargo build --release

# Build DBCop
RUN cargo build --release --manifest-path tools/dbcop/Cargo.toml


FROM ubuntu:24.04

RUN apt-get update && apt-get install -y python3 python3-pip python-is-python3

WORKDIR /awdit
RUN mkdir -p target/release
RUN mkdir -p tools/dbcop/target/release
COPY --from=build /awdit/target/release/awdit target/release/awdit
COPY --from=build /awdit/tools/dbcop/target/release/dbcop tools/dbcop/target/release/dbcop
COPY tools tools/
COPY scripts scripts/

# Build tools
RUN cd tools/CausalC+ && pip install --break-system-packages -r requirements.txt
RUN cd tools/mono && pip install --break-system-packages -r requirements.txt

ENTRYPOINT ["/bin/bash"]
