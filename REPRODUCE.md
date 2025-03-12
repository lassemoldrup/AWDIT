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
docker run -ti -v $(pwd)/results:/awdit/results -v $(pwd)/res:/awdit/res awdit-artifact
```

## Reproduce a Figure

To reproduce a single figure, e.g. figure 7, run the Docker image as above, and run

```shell
scripts/reproduce-fig7.sh
```

The result will be in `results/fig7`.

## Reproduce all Figures

To reproduce all figures, run the Docker image as above, and run

```shell
scripts/reproduce-all.sh
```

Note: this will take approximately two days.
