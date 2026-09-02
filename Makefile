.PHONY: test smoke install bootstrap stream-build

test:
	python3 -m unittest discover -s tests -v

smoke:
	./scripts/smoke_test.sh

install:
	./scripts/install.sh

bootstrap:
	./scripts/bootstrap_macos.sh

stream-build:
	./scripts/build_streaming_macos.sh
