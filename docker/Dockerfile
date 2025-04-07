FROM rust:1.85 AS build

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    libclang-dev llvm

WORKDIR /awdit
COPY Cargo.toml Cargo.lock build.rs ./
COPY src src/
COPY tools/dbcop tools/dbcop/

# Build AWDIT
RUN cargo build --release --features dbcop

# Build DBCop
RUN cargo build --release --manifest-path tools/dbcop/Cargo.toml


FROM ubuntu:24.04

# Install dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y software-properties-common; \
    add-apt-repository ppa:sosy-lab/benchmarking; \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python-is-python3 openjdk-21-jre benchexec git g++ make cmake zlib1g-dev libgmp3-dev fuse-overlayfs texlive-latex-base
RUN rm -rf /var/lib/apt/lists/*

# Install Monosat
# RUN git clone https://github.com/sambayless/monosat.git
# RUN cd monosat && git checkout fa810d4 && cmake -DPYTHON=ON . && make && make install
# RUN rm -rf monosat

# Install LaTeX packages
RUN tlmgr install xcolor tikz pgfplots

# Create non-root user
RUN useradd -ms /bin/bash user
WORKDIR /home/user/awdit

# Copy files
COPY tools tools/
RUN rm -rf tools/dbcop
RUN cd tools/CausalC+ && pip install --break-system-packages -r requirements.txt
RUN cd tools/mono && pip install --break-system-packages -r requirements.txt
COPY --from=build /awdit/tools/dbcop/target/release/dbcop tools/dbcop/target/release/dbcop
COPY --from=build /awdit/target/release/awdit target/release/awdit
COPY scripts scripts/
COPY results/graphs results/graphs
COPY docker/init-runexec.sh /init-runexec.sh

RUN chown -R user .
# Set non-root user as default
USER user

# Create output directory for DBCop
RUN mkdir -p /home/user/awdit/dbcop-out

# Create volume for results
RUN mkdir -p /home/user/awdit/results
VOLUME /home/user/awdit/results

ENTRYPOINT [ "/init-runexec.sh" ]
CMD [ "bash" ]