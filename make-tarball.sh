#!/bin/bash
set -eu

rm -r AWDIT 2> /dev/null || true
mkdir AWDIT
cp -R docker AWDIT
cp -R scripts AWDIT
cp -R src AWDIT
cp -R results AWDIT
cp -R tools AWDIT
cp Cargo.toml Cargo.lock build.rs README.md rustfmt.toml AWDIT

docker buildx build --platform linux/arm64 \
-o type=docker,dest=AWDIT/docker/awdit-artifact-arm64.tar.gz,compression=gzip \
-f docker/Dockerfile -t awdit-artifact .

docker buildx build --platform linux/amd64 \
-o type=docker,dest=AWDIT/docker/awdit-artifact-amd64.tar.gz,compression=gzip \
-f docker/Dockerfile -t awdit-artifact .

chmod -R a+rwX AWDIT
tar -cJf AWDIT.tar.xz --no-xattrs --uid 0 --gid 0 AWDIT
