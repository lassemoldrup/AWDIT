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

docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t awdit-artifact .
docker save -o AWDIT/docker/awdit-artifact.tar.gz awdit-artifact

chmod -R u+rwX,a+rX AWDIT
tar -cJf AWDIT.tar.xz --uid 0 --gid 0 AWDIT
