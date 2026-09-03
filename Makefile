.PHONY: doctor bootstrap test install redmetal native

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

native:
	bash ./scripts/build_native.sh
	bash ./scripts/build_layer_audit.sh
	bash ./scripts/build_deltanet_proj.sh
	bash ./scripts/build_deltanet_prestate.sh
	bash ./scripts/build_deltanet_state.sh
	bash ./scripts/build_deltanet_tail.sh
	bash ./scripts/build_deltanet_layer.sh
	bash ./scripts/build_recurrent_block.sh
	bash ./scripts/build_attention.sh
	bash ./scripts/build_attention_block.sh
