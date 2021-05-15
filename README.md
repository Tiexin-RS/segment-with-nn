# segment-with-nn
segmentation using nerual network way

## Development

### Prerequirements
1. Make sure there's `docker-compose` & `docker` on your machine.

2. Create `.env` file with content as:
    ``` bash
    UID=1002
    GID=1002
    DATA_DIR=/path/to/dataset
	TVM_DIR=/path/to/tvm
    ```

### Start dev containers!
Execute as follows:
```bash
# if your want to start service
DC_PREFIX=<prefix-for-differ> make dev

# if your want to force rebuild the image
DC_PREFIX=<prefix-for-differ> make dev-build

# if your want to attach to some pod
DC_PREFIX=<prefix-for-differ> POD_NAME=<pod_name> make attach
```

