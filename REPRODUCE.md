# Reproducing Our Findings

## Prerequisites

The experiments require
- An (x86 or ARM) Linux system with kernel version >= 4.5
- Docker

We also provide VirtualBox images matching the above requirements.
However, VMs do not produce reliable performance results, so please only use this as a last resort. 

## Running the Docker Image

To run the Docker image do the following.

### 1

Load the image:

```shell
docker load -i docker/image.tar.gz
```

Optionally, you can pass a platform (`linux/amd64` or `linux/arm64`, depending on your system):

```shell
docker load -i docker/image.tar.gz --platform linux/amd64
```

### 2

Run the image:

```shell
docker run --name awdit-container --privileged --cap-drop=all -ti -v $(pwd)/results:/home/user/awdit/results -v $(pwd)/histories:/home/user/awdit/histories awdit-artifact
```

**Note:** The reason for `--privileged` is that we use [BenchExec](https://github.com/sosy-lab/benchexec) for running benchmarks, which uses Linux cgroups. To mitigate any risks when running like this, we use `--cap-drop=all` to disable as many capabilities of the container a possible, and the active user in the Docker container will be non-root. See [this document](https://github.com/sosy-lab/benchexec/blob/26ea602ced1bf339db124efb3cb53bc5dd94098c/doc/benchexec-in-container.md) for more information.

**Note:** To run the container more than once, run the following to delete the old container:

```shell
docker rm -v awdit-container
```

## Reproducing a Figure

To reproduce a figure, e.g. Figure 7, run the Docker image as above, and inside run

```shell
scripts/reproduce-fig7.sh
```

By default, this runs with the time and memory limits used in the paper.
To run the experiments with different constraints, e.g. 10 second timeout and 25 GiB memory limit, run

```shell
TIME_LIMIT=10 MEM_LIMIT=25 scripts/reproduce-fig7.sh
```

After the shell scripts finishes, exit the docker image, and run

```shell
docker cp awdit-container:/home/user/awdit/results .
```

The result will be in `results/fig7`.

The approximate running time and memory usage for reproducing each figure is shown below:

| Figure | Time (defaults) | Time (10 s, 25 GiB) | Memory (defaults) |
| :----- | :-------------- | ------------------: | ----------------: |
| fig7   | 5 hours         |              10 min |            50 GiB |
| fig8   | 48 hours        |              65 min |            50 GiB |
| fig9   | 6 min           |               6 min |           2.3 GiB |

## Reproducing all Figures

To reproduce all figures, run the Docker image as above, and run

```shell
scripts/reproduce-all.sh
```

As above, this can also be done with different time/memory constraints, e.g.

```shell
TIME_LIMIT=10 MEM_LIMIT=25 scripts/reproduce-all.sh
```

After the shell scripts finishes, exit the docker image, and run

```shell
docker cp awdit-container:/home/user/awdit/results .
```
