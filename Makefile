.PHONY: doctor bootstrap test install redmetal

doctor:
	PYTHONPATH=. python3 -m redlite.cli doctor

bootstrap:
	./scripts/bootstrap_macos.sh

test:
	PYTHONPATH=. python3 -m unittest discover -s tests -v

install:
	./scripts/install.sh

redmetal:
	bash ./scripts/build_redmetal.sh
