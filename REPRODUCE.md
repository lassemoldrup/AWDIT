# Reproducing Our Findings

To run the Docker image, do the following:

### 1.

Install Docker.

### 2.

Load the image:

```shell
docker load -i docker/image.tar.gz
```

Optionally, you can pass a platform (`linux/amd64` or `linux/arm64`, depending on your system):

```shell
docker load -i docker/image.tar.gz --platform linux/amd64
```

### 3.

Run the image:

```shell
docker run --name awdit-container --privileged --cap-drop=all -ti -v $(pwd)/results:/home/user/awdit/results -v $(pwd)/histories:/home/user/awdit/histories awdit-artifact
```

**Note:** The reason for `--privileged` is that we use [BenchExec](https://github.com/sosy-lab/benchexec) for running benchmarks, which uses Linux cgroups. See [this document](https://github.com/sosy-lab/benchexec/blob/26ea602ced1bf339db124efb3cb53bc5dd94098c/doc/benchexec-in-container.md) for more information.

**Note:** To run the container more than once, run the following to delete the old container:
```shell
docker rm -v awdit-container
```

## Reproduce a Figure

To reproduce a single figure, e.g. figure 7, run the Docker image as above, and inside run

```shell
scripts/reproduce-fig7.sh
```

After the shell scripts finishes, exit the docker image, and run

```shell
docker cp awdit-container:/home/user/awdit/results .
```

The result will be in `results/fig7`.

## Reproduce all Figures

To reproduce all figures, run the Docker image as above, and run

```shell
scripts/reproduce-all.sh
```

After the shell scripts finishes, exit the docker image, and run

```shell
docker cp awdit-container:/home/user/awdit/results .
```

**Note:** this will take approximately two days.
