import pytest
import yaml

from bialy import config

EXAMPLES = sorted((config.ROOT / "config" / "runs").glob("*.example.yaml"))


def test_examples_exist():
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_run_sections_load_and_name_known_inputs(path, factory):
    run = config.run_settings(yaml.safe_load(path.read_text(encoding="utf-8"))["run"])
    archetypes = {a.id for a in factory.archetypes}
    for name in run["workspaces"]:
        factory.workspace(name)
    assert set(run["archetypes"]) <= archetypes
    for model in [*run["generators"], *run["judges"]]:
        factory.model(model)
    student = run["custom_model"]
    factory.model(student["student"])
    factory.model(student["teacher"])
