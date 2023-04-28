build-m3gnet:
	docker build -t m3gnet ./ctx/m3gnet

train-m3gnet:
	docker run --rm \
	--mount type=bind,source="$(shell pwd)"/data,target=/app/data \
	--mount type=bind,source="$(shell pwd)"/utils,target=/app/utils \
	--mount type=bind,source="$(shell pwd)"/results,target=/app/results \
	m3gnet $(ARGFILE)\

build-schnet:
	docker build -t schnet ./ctx/schnet

train-schnet:
	docker run --rm \
	--mount type=bind,source="$(shell pwd)"/data,target=/app/data \
	--mount type=bind,source="$(shell pwd)"/utils,target=/app/utils \
	--mount type=bind,source="$(shell pwd)"/results,target=/app/results \
	schnet $(ARGFILE)\

build-dimenet:
	docker build -t dimenet ./ctx/dimenet

train-dimenet:
	docker run --rm \
	--mount type=bind,source="$(shell pwd)"/data,target=/app/data \
	--mount type=bind,source="$(shell pwd)"/utils,target=/app/utils \
	--mount type=bind,source="$(shell pwd)"/results,target=/app/results \
	dimenet $(ARGFILE)\