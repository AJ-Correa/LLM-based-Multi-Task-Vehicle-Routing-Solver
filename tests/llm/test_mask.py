import json
import random
import time

import pytest

from vrp_service.contracts import TRAINING_VARIANTS
from vrp_service.env import check_routes, load_instance
from vrp_service.eval import load_instance_set, read_request
from vrp_service.llm.mask import RouteMask
from vrp_service.llm.parse import parse_routes
from vrp_service.serving.settings import project_root

ROOT = project_root()
INSTANCE_SET = load_instance_set(ROOT / "configs/eval/instance_set_v1.yaml", ROOT)
EOS = 0
# Single characters plus merged pieces like a BPE vocabulary has, and junk that never fits.
PIECES = [*'{"routes:[],}0123456789', '{"', "routes", '":[[', "],[", "]]}", "12", "5,", " ", "a"]
VOCAB = [(token, piece) for token, piece in enumerate(PIECES, start=1)]


def _decode(variant: str, seed: int) -> str:
    request = read_request(INSTANCE_SET, variant, 0, backend="llm")
    mask = RouteMask(load_instance(request), VOCAB, EOS)
    pieces = dict(VOCAB)
    rng = random.Random(seed)
    text = ""
    while (allowed := mask.allowed(text)) != [EOS]:
        assert allowed, text
        text += pieces[rng.choice(allowed)]
    return text


@pytest.mark.parametrize("variant", TRAINING_VARIANTS)
def test_any_masked_answer_is_feasible(variant: str) -> None:
    request = read_request(INSTANCE_SET, variant, 0, backend="llm")
    for seed in range(3):
        text = _decode(variant, seed)
        assert parse_routes(text, request).violations == [], text


def test_repeats_unknown_ids_depots_and_early_endings_are_banned() -> None:
    request = read_request(INSTANCE_SET, "cvrp", 0, backend="llm")
    mask = RouteMask(load_instance(request), VOCAB, EOS)
    token = {piece: token for token, piece in VOCAB}
    assert mask.allowed("") == [token["{"], token['{"']]
    assert token["0"] not in mask.allowed('{"routes":[[')
    assert token["]"] not in mask.allowed('{"routes":[[')  # empty route
    # After customer 1, "1" may still start 10-19, but cannot end as 1 again.
    assert token["1"] in mask.allowed('{"routes":[[1,')
    assert token[","] not in mask.allowed('{"routes":[[1,1')
    assert token["a"] not in mask.allowed('{"routes":[[1,')
    assert token["]]}"] not in mask.allowed('{"routes":[[1')  # 99 customers unvisited


def test_a_known_feasible_answer_passes_the_mask() -> None:
    request = read_request(INSTANCE_SET, "cvrp", 0, backend="llm")
    instance = load_instance(request)
    mask = RouteMask(instance, VOCAB, EOS)
    text = _decode("cvrp", 7)
    routes = json.loads(text)["routes"]
    assert not check_routes(instance, [[0, *r, 0] for r in routes])
    started = time.perf_counter()
    for end in range(len(text)):
        assert mask.allowed(text[:end])
    assert mask.allowed(text) == [EOS]
    assert time.perf_counter() - started < 5
