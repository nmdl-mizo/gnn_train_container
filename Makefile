build:
	docker build -t $(MODEL) ./ctx/$(MODEL)

train:
	docker run --rm \
	-u $(shell id -u):$(shell id -g) \
	--env-file .env \
	--runtime nvidia \
	--shm-size=16g \
	--mount type=bind,source="$(shell pwd)"/data,target=/app/data \
	--mount type=bind,source="$(shell pwd)"/common,target=/app/common \
	--mount type=bind,source="$(shell pwd)"/results,target=/app/results \
	$(MODEL) $(ARG) \

inspect:
	docker run --rm -it \
	-u $(shell id -u):$(shell id -g) \
	--env-file .env \
	--runtime nvidia \
	--shm-size=1g \
	--mount type=bind,source="$(shell pwd)"/data,target=/app/data \
	--mount type=bind,source="$(shell pwd)"/common,target=/app/common \
	--mount type=bind,source="$(shell pwd)"/results,target=/app/results \
	--entrypoint bash \
	$(MODEL) \
