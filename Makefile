build:
	docker build -t $(MODEL) ./ctx/$(MODEL)

train:
	docker run --rm \
	-u $(id -u):$(id -g) \
	--env-file .env \
	--runtime nvidia \
	--mount type=bind,source="$(shell pwd)"/data,target=/app/data \
	--mount type=bind,source="$(shell pwd)"/common,target=/app/common \
	--mount type=bind,source="$(shell pwd)"/results,target=/app/results \
	$(MODEL) $(ARGFILE) \

inspect:
	docker run --rm -it \
	-u $(shell id -u):$(shell id -g) \
	--mount type=bind,source="$(shell pwd)"/data,target=/app/data \
	--mount type=bind,source="$(shell pwd)"/common,target=/app/common \
	--mount type=bind,source="$(shell pwd)"/results,target=/app/results \
	--entrypoint bash \
	$(MODEL) \
