"""CPU smoke test: importing TrainPilot must succeed without a GPU."""


def test_import_trainpilot() -> None:
    import trainpilot

    assert trainpilot.__version__
