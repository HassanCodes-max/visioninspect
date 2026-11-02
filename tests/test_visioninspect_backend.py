import importlib.util
import io
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).parent.parent / "visioninspect"
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "probes"))
import finetune  # noqa: E402
from app import main  # noqa: E402
from app.config import PRODUCT_NAMES, Settings  # noqa: E402
from app.pipelines import batch, explain, quality, segment, vision  # noqa: E402

spec = importlib.util.spec_from_file_location("vi_make_bands", ROOT / "backend" / "make_bands.py")
make_bands = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_bands)

BANDS = quality.Bands(low=0.2, high=0.6, pass_good_share=0.97, reject_defect_share=0.95, review_defect_share=0.5)


def tiny_backbone():
    return transformers.Dinov2Model(transformers.Dinov2Config(hidden_size=24, num_hidden_layers=2, num_attention_heads=2,
                                                              mlp_ratio=2, image_size=448, patch_size=14))


def jpeg():
    stream = io.BytesIO()
    Image.new("RGB", (640, 480), (120, 80, 40)).save(stream, "JPEG")
    return stream.getvalue()


class FakeRecogniser:
    def __init__(self, supported):
        self.answer = vision.Recognition("chewinggum", 0.9, supported)

    def check(self, image):
        return self.answer


class FakeFinder:
    def __init__(self, score):
        heat = np.zeros(quality.PATCHES, dtype=np.float32)
        heat[:3] = 0.9
        self.found = vision.Finding(score, heat)

    def inspect(self, image):
        return self.found


def client(score=0.9, supported=True):
    settings = Settings()
    service = main.Service(settings, FakeRecogniser(supported), FakeFinder(score), BANDS)
    return TestClient(main.create_app(service, settings)), settings


def post(api, data=None, name="a.jpg"):
    return api.post("/api/inspect", files={"photo": (name, data if data is not None else jpeg(), "image/jpeg")})


def test_decisions_follow_the_bands():
    assert quality.decide(0.7, True, BANDS)[0] == quality.REJECT
    assert quality.decide(0.1, True, BANDS)[0] == quality.PASS
    assert quality.decide(0.4, True, BANDS)[0] == quality.REVIEW


def test_unsupported_items_are_never_passed_or_rejected():
    assert quality.decide(0.0, False, BANDS) == (quality.REVIEW, "not_supported")
    assert quality.decide(0.99, False, BANDS) == (quality.REVIEW, "not_supported")


def test_regions_are_separate_and_measured():
    heat = np.zeros((32, 32))
    heat[0, 0:2] = 1
    heat[10:12, 10:12] = 1
    regions = quality.find_regions(heat, 0.5)
    assert len(regions) == 2
    assert regions[0]["area_percent"] == round(100 * 4 / 1024, 2)
    assert regions[0]["x"] == 10 / 32 and regions[0]["width"] == 2 / 32


def test_clean_heat_map_has_no_regions():
    assert quality.find_regions(np.zeros(1024), 0.5) == []


def test_network_matches_what_training_saved():
    trained = finetune.Inspector(tiny_backbone(), 24)
    network = vision.DefectNet(tiny_backbone(), 24)
    network.load_state_dict(trained.state_dict())


def test_weights_saved_under_other_attention_names_still_load():
    network = vision.DefectNet(tiny_backbone(), 24)
    wanted = network.state_dict()
    older = {key.replace(new, old): value for key, value in wanted.items()
             for old, new in vision.ATTENTION_NAMES.items() if new in key}
    older.update({key: value for key, value in wanted.items() if ".attention." not in key})
    assert set(older) != set(wanted)
    network.load_state_dict(vision.fit_names(older, wanted))


def test_photos_reach_the_model_the_way_training_read_them():
    pixels = np.random.default_rng(0).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    photo = Image.fromarray(pixels)
    stream = io.BytesIO()
    photo.resize((448, 448), Image.BILINEAR).save(stream, "JPEG", quality=95)
    assert np.array_equal(np.asarray(vision.as_trained(photo)), np.asarray(Image.open(stream)))


def test_finder_gives_a_score_and_a_full_map():
    found = vision.DefectFinder(vision.DefectNet(tiny_backbone(), 24)).inspect(Image.new("RGB", (300, 200), "white"))
    assert 0 <= found.score <= 1
    assert found.heat.shape == (1024,)


def test_recogniser_needs_closeness_and_a_listed_product():
    gallery = np.eye(8, dtype=np.float32)
    names = np.array(["candle"] * 6 + ["pcb3"] * 2)
    recogniser = vision.Recogniser(tiny_backbone(), gallery, names, 0.1, PRODUCT_NAMES)
    near = np.ones(8, dtype=np.float32) / np.sqrt(8)
    assert recogniser.match(near).closeness > 0.1
    listed = np.zeros(8, dtype=np.float32)
    listed[0] = 1
    assert recogniser.match(listed).supported
    left_off = np.zeros(8, dtype=np.float32)
    left_off[7] = 1
    match = recogniser.match(left_off)
    assert match.product == "pcb3" and not match.supported


def test_pcb3_and_screw_are_left_off_the_list():
    assert "pcb3" not in PRODUCT_NAMES and "mvtec_screw" not in PRODUCT_NAMES
    assert len(PRODUCT_NAMES) == 25


def test_api_rejects_with_a_map():
    api, _ = client(score=0.9)
    answer = post(api).json()
    assert answer["decision"] == "REJECT" and answer["product"] == "Chewing gum"
    assert len(answer["heat"]) == 1024 and answer["regions"] and answer["tested_share"] == 0.95


def test_api_passes_a_clean_item():
    assert post(client(score=0.05)[0]).json()["decision"] == "PASS"


def test_api_sends_unsupported_items_to_review_without_scoring():
    answer = post(client(supported=False)[0]).json()
    assert answer["decision"] == "REVIEW" and answer["reason"] == "not_supported"
    assert answer["product"] is None and answer["score"] is None and answer["heat"] is None


def test_api_refuses_a_file_that_is_not_a_photo():
    response = post(client()[0], b"not a photo", "a.txt")
    assert response.status_code == 415


def test_api_refuses_a_large_file():
    settings = replace(Settings(), max_upload_bytes=1000)
    service = main.Service(settings, FakeRecogniser(True), FakeFinder(0.9), BANDS)
    assert post(TestClient(main.create_app(service, settings))).status_code == 413


def test_cpu_check_still_decides_listed_products():
    checker = main.Service(Settings(), FakeRecogniser(True), FakeFinder(0.9), BANDS)
    answer = main.on_cpu(checker, [Image.new("RGB", (64, 64))] * 2)
    assert answer["fallback"] and [r["decision"] for r in answer["results"]] == [quality.REJECT] * 2
    assert answer["results"][0]["outline"] is not None and answer["analysis"]["rejected"] == 2


def test_cpu_check_tells_other_items_the_ai_is_resting():
    checker = main.Service(Settings(), FakeRecogniser(False), FakeFinder(0.9), BANDS)
    result = main.on_cpu(checker, [Image.new("RGB", (64, 64))])["results"][0]
    assert (result["decision"], result["reason"]) == (quality.REVIEW, "ai_resting")


def test_batch_route_says_when_it_ran_on_the_cpu():
    settings = Settings()

    class Resting(main.Service):
        def inspect_many(self, images, request=None, limit=5.0, stock=None):
            return main.on_cpu(main.Service(settings, FakeRecogniser(True), FakeFinder(0.9), BANDS), images, limit, stock)

    api = TestClient(main.create_app(Resting(settings), settings))
    reply = api.post("/api/inspect-batch", files=[("photos", ("a.jpg", jpeg(), "image/jpeg"))]).json()
    assert reply["fallback"] is True and reply["results"][0]["decision"] == quality.REJECT
    normal, _ = client()
    assert normal.post("/api/inspect-batch", files=[("photos", ("a.jpg", jpeg(), "image/jpeg"))]).json()["fallback"] is False


def test_page_tags_styles_and_scripts_so_updates_always_load():
    api, _ = client()
    page = api.get("/")
    assert page.headers["cache-control"] == "no-cache"
    assert "/static/app.js?v=" in page.text and "/static/styles.css?v=" in page.text


def test_nobody_is_signed_in_without_a_session():
    api, _ = client()
    assert api.get("/api/me").json() == {"signed_in": False}


def test_signed_in_visitor_is_named_from_the_session():
    _, settings = client()
    app = main.create_app(main.Service(settings, FakeRecogniser(True), FakeFinder(0.9), BANDS), settings)

    async def with_session(scope, receive, send):
        scope["session"] = {"oauth_info": {"userinfo": {"name": "Hassan Ali", "preferred_username": "hassan"}}}
        await app(scope, receive, send)

    assert TestClient(with_session).get("/api/me").json() == {"signed_in": True, "name": "Hassan Ali", "username": "hassan"}


def test_api_lists_supported_products():
    products = client()[0].get("/api/products").json()["products"]
    assert "Chewing gum" in products and len(products) == 25


def test_batch_checks_good_photos_and_reports_bad_ones():
    api, _ = client()
    files = [("photos", ("a.jpg", jpeg(), "image/jpeg")), ("photos", ("b.jpg", b"not a photo", "image/jpeg")),
             ("photos", ("c.jpg", jpeg(), "image/jpeg"))]
    results = api.post("/api/inspect-batch", files=files).json()["results"]
    assert [r.get("decision", r.get("error")) for r in results] == ["REJECT", "notImage", "REJECT"]


def test_batch_uses_one_call_to_the_models():
    settings = Settings()
    calls = []

    class Counted(main.Service):
        def inspect_many(self, images, *rest):
            calls.append(len(images))
            return super().inspect_many(images, *rest)

    api = TestClient(main.create_app(Counted(settings, FakeRecogniser(True), FakeFinder(0.9), BANDS), settings))
    api.post("/api/inspect-batch", files=[("photos", ("a.jpg", jpeg(), "image/jpeg"))] * 3)
    assert calls == [3]


def test_batch_refuses_too_many_photos():
    api, settings = client()
    response = api.post("/api/inspect-batch", files=[("photos", ("a.jpg", b"x", "image/jpeg"))] * (settings.max_batch + 1))
    assert response.status_code == 413


class FakeExplainer:
    def __init__(self, verdict):
        self.calls = 0
        defective = verdict == "defect"
        self.answer = {"item": "bag", "verdict": verdict, "defect": "tear" if defective else "", "where": "left" if defective else "",
                       "region": explain.to_region([100, 100, 300, 300]) if defective else None,
                       "severity": "major" if defective else "", "fix": "Patch it." if defective else ""}

    def explain(self, image, region=None, flagged=False):
        self.calls += 1
        self.asked = (region, flagged)
        return self.answer


class FakeSegmenter:
    def __init__(self):
        self.calls = 0

    def outline(self, image, region):
        self.calls += 1
        return {"mask": "ab", "area_percent": 3.5}


def service(score, supported, verdict, segmenter=None):
    helper = FakeExplainer(verdict)
    return main.Service(Settings(), FakeRecogniser(supported), FakeFinder(score), BANDS, helper, segmenter), helper


def test_a_failed_ai_summary_keeps_the_batch_results():
    class Broken(FakeExplainer):
        def advise(self, facts):
            raise RuntimeError("out of memory")

    checker = main.Service(Settings(), FakeRecogniser(False), FakeFinder(0.0), BANDS, Broken("defect"))
    answer = checker.inspect_many([Image.new("RGB", (64, 64))] * 2)
    assert answer["analysis"]["rejected"] == 2
    assert answer["analysis"]["causes"] == []


def test_mask_survives_packing_for_the_browser():
    mask = np.zeros((segment.GRID, segment.GRID), dtype=bool)
    mask[10:30, 40:90] = True
    assert np.array_equal(segment.unpack_mask(segment.pack_mask(mask)), mask)


def test_outline_measures_the_full_size_mask():
    mask = np.zeros((400, 800), dtype=bool)
    mask[:100, :200] = True
    assert segment.to_outline(mask)["area_percent"] == 6.25


def test_an_empty_mask_gives_no_outline():
    assert segment.to_outline(np.zeros((50, 50), dtype=bool)) is None


def test_ai_defect_gets_an_outline_and_its_real_area():
    outlines = FakeSegmenter()
    helper_service, _ = service(0.0, False, "defect", outlines)
    answer = helper_service.inspect(Image.new("RGB", (64, 64)))
    assert (answer["outline"], answer["affected_percent"]) == ("ab", 3.5)


def test_listed_products_are_outlined_from_their_own_map_without_sam():
    outlines = FakeSegmenter()
    helper_service, _ = service(0.9, True, "defect", outlines)
    answer = helper_service.inspect(Image.new("RGB", (64, 64)))
    mask = segment.unpack_mask(answer["outline"])
    assert outlines.calls == 0 and mask[0, :20].all() and not mask[64:].any()
    assert answer["affected_percent"] == round(100 * float(mask.mean()), 2)


def test_passed_products_get_no_outline():
    helper_service, _ = service(0.1, True, "good")
    answer = helper_service.inspect(Image.new("RGB", (64, 64)))
    assert answer["decision"] == "PASS" and answer["outline"] is None


def test_listed_fault_with_nothing_on_the_map_uses_the_ai_outline():
    finder = FakeFinder(0.4)
    finder.found = vision.Finding(0.4, np.zeros(quality.PATCHES, dtype=np.float32))
    outlines = FakeSegmenter()
    checker = main.Service(Settings(), FakeRecogniser(True), finder, BANDS, FakeExplainer("defect"), outlines)
    answer = checker.inspect(Image.new("RGB", (64, 64)))
    assert answer["decision"] == "REVIEW" and len(answer["regions"]) == 1
    assert (answer["outline"], answer["affected_percent"]) == ("ab", 3.5)


def test_map_outline_is_smooth_and_empty_maps_give_none():
    heat = np.zeros(quality.PATCHES, dtype=np.float32)
    assert segment.from_heat(heat, 0.5) is None
    heat[quality.GRID + 1] = 0.9
    widths = segment.unpack_mask(segment.from_heat(heat, 0.5)["mask"]).sum(axis=1)
    assert len(set(widths.tolist())) > 2


def test_ai_reply_is_read_with_its_marked_area():
    reply = 'Here: ```json\n{"item": "leather", "verdict": "defect", "defect": "hole", "where": "left", "box": [0, 150, 450, 650], "severity": "major", "fix": "Patch it."}\n```'
    found = explain.parse_reply(reply)
    assert (found["verdict"], found["defect"], found["severity"]) == ("defect", "hole", "major")
    assert found["region"]["width"] == pytest.approx(0.45)
    assert found["region"]["area_percent"] == pytest.approx(22.5)


@pytest.mark.parametrize("reply", ["no json here", '{"verdict": "broken"}', '{"verdict": '])
def test_unreadable_ai_replies_become_unsure(reply):
    found = explain.parse_reply(reply)
    assert found["verdict"] == "unsure"
    assert found["region"] is None


@pytest.mark.parametrize("box", [[], [1, 2, 3], [500, 500, 100, 100], ["a", 0, 1, 2], None])
def test_unusable_boxes_are_dropped(box):
    assert explain.to_region(box) is None


def test_a_good_reply_keeps_no_defect_details():
    found = explain.parse_reply('{"verdict": "good", "defect": "scratch", "box": [1, 2, 3, 4], "fix": "x"}')
    assert (found["defect"], found["region"], found["fix"]) == ("", None, "")


def test_unlisted_item_with_a_defect_is_rejected_and_marked():
    helper_service, _ = service(0.0, False, "defect")
    answer = helper_service.inspect(Image.new("RGB", (64, 64)))
    assert (answer["decision"], answer["reason"]) == (quality.REJECT, "ai_defect")
    assert len(answer["regions"]) == 1 and answer["heat"] is None


def test_unlisted_item_the_ai_sees_as_good_passes_with_its_tested_share():
    helper_service, _ = service(0.0, False, "good")
    answer = helper_service.inspect(Image.new("RGB", (64, 64)))
    assert (answer["decision"], answer["reason"], answer["tested_share"]) == (quality.PASS, "ai_good", quality.AI_EVIDENCE[quality.PASS])


def test_unsure_ai_answer_goes_to_review():
    helper_service, _ = service(0.0, False, "unsure")
    assert helper_service.inspect(Image.new("RGB", (64, 64)))["decision"] == quality.REVIEW


def test_rejected_listed_item_tells_the_ai_where_the_defect_is():
    helper_service, helper = service(0.9, True, "defect")
    helper_service.inspect(Image.new("RGB", (64, 64)))
    region, flagged = helper.asked
    assert flagged and region["x"] == 0 and region["y"] == 0


def test_hint_box_uses_the_model_scale():
    assert explain.to_box({"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}) == [100, 200, 400, 600]


def test_listed_item_keeps_the_measured_decision_and_gets_advice():
    helper_service, _ = service(0.9, True, "good")
    answer = helper_service.inspect(Image.new("RGB", (64, 64)))
    assert answer["decision"] == quality.REJECT
    assert answer["explanation"]["item"] == "bag"


def test_clean_listed_item_uses_no_ai_time():
    helper_service, helper = service(0.05, True, "good")
    assert helper_service.inspect(Image.new("RGB", (64, 64)))["decision"] == quality.PASS
    assert helper.calls == 0


def test_without_the_ai_unlisted_items_still_go_to_review():
    api, _ = client(supported=False)
    assert post(api).json()["reason"] == "not_supported"


def rows(decisions, defect="tear", severity="major"):
    found = {"defect": defect, "severity": severity}
    return [{"decision": d, "explanation": found if d == quality.REJECT else None, "affected_percent": 2.0 if d == quality.REJECT else 0.0} for d in decisions]


def test_ten_photos_with_three_rejected_give_thirty_percent():
    stats = batch.analyse(rows([quality.REJECT] * 3 + [quality.PASS] * 7))
    assert (stats["photos"], stats["rejected"], stats["defect_rate"]) == (10, 3, 30.0)
    assert stats["range_low"] == pytest.approx(10.8, abs=0.1)
    assert stats["range_high"] == pytest.approx(60.3, abs=0.1)


def test_most_common_defects_come_first_with_shares():
    items = rows([quality.REJECT] * 3, "tear") + rows([quality.REJECT], "stain")
    pareto = batch.analyse(items)["pareto"]
    assert [(p["defect"], p["share"], p["running_share"]) for p in pareto] == [("tear", 75, 75), ("stain", 25, 100)]


@pytest.mark.parametrize("rejected, review, total, limit, advice", [
    (5, 0, 10, 5, "hold"), (0, 0, 100, 5, "accept"), (0, 0, 10, 5, "check_more"), (0, 4, 100, 5, "check_more")])
def test_lot_advice_allows_for_sample_size_and_unchecked_photos(rejected, review, total, limit, advice):
    assert batch.lot_advice(rejected, review, total, limit) == advice


def test_stock_estimate_scales_the_sample():
    assert batch.stock_estimate(3, 10, 500) == {"stock": 500, "likely": 150, "low": 54, "high": 302}


def test_stock_estimate_needs_a_stock_size():
    assert batch.stock_estimate(3, 10, None) is None


def test_ai_advice_with_numbers_is_dropped():
    reply = '{"causes": ["Rough handling", "About 40% from one supplier"], "actions": ["Check packing", 5]}'
    assert batch.parse_advice(reply) == {"causes": ["Rough handling"], "actions": ["Check packing"]}


def test_unreadable_ai_advice_gives_nothing():
    assert batch.parse_advice("sorry") == {"causes": [], "actions": []}


def test_batch_route_returns_the_analysis_and_uses_the_limit():
    api, _ = client()
    files = [("photos", ("a.jpg", jpeg(), "image/jpeg"))] * 4
    analysis = api.post("/api/inspect-batch", files=files, data={"limit": "10", "stock": "200"}).json()["analysis"]
    assert (analysis["photos"], analysis["limit"], analysis["stock"]["likely"]) == (4, 10.0, 200)


def test_a_batch_with_only_bad_files_has_no_analysis():
    api, _ = client()
    reply = api.post("/api/inspect-batch", files=[("photos", ("a.jpg", b"nope", "image/jpeg"))]).json()
    assert reply["analysis"] is None


def test_photo_sideways_in_exif_is_turned_upright():
    stream = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (40, 20)).save(stream, "JPEG", exif=exif)
    assert main.read_photo(stream.getvalue(), Settings()).size == (20, 40)


def test_bands_leave_out_low_products_and_report_real_shares():
    rows = [{"product": p, "label": label, "score": s} for p in ("a", "b", "c", "pcb3") for label, s in
            ((0, 0.1), (0, 0.15), (0, 0.2), (1, 0.8), (1, 0.85), (1, 0.9))]
    result = make_bands.measure(rows)
    assert result["tested"]["images"] == 18
    assert result["bands"]["low"] == 0.2 and result["bands"]["high"] == 0.8
    assert result["tested"]["correct"] == 1
