from transcript_anonymizer.detection import Detector, load_policy
from transcript_anonymizer.selftest import run


def test_selftest_workflow_in_rules_mode(tmp_path):
    result = run(tmp_path / 'self-test', detector=Detector(load_policy(None), rules_only=True))
    assert result['ok'] and not result['model_enabled']
