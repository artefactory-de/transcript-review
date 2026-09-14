"""Synthetic regressions for business terms, birth context and local name variants."""
import re

import pytest

from transcript_anonymizer.detection import Detector, load_policy


class Tokenizer:
    model_max_length = 512

    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class Predictions:
    tokenizer = Tokenizer()

    def __init__(self, label, value):
        self.label, self.value = label, value

    def extract_entities(self, text, labels, **kwargs):
        return {'entities': {self.label: [
            {'start': m.start(), 'end': m.end(), 'text': m.group(), 'score': 0.99}
            for m in re.finditer(re.escape(self.value), text)
        ]}}


@pytest.mark.parametrize('label', ['DAX-Konzern', 'Account Manager', 'Leitender Sachbearbeiter',
                                  'Business Analyst', 'SAP', 'Microsoft Teams', 'Kundenberaterin'])
def test_non_person_terms_are_not_promoted_even_by_confident_model(label, tmp_path):
    detector = Detector(load_policy(None), tmp_path)
    detector._model = Predictions('full_name', label)
    assert detector.detect([{'id': 's1', 'text': f'{label}: Der Vorgang ist abgeschlossen.'}]) == []


@pytest.mark.parametrize('text,value,expected', [
    ('[00:01:23] Anna Müller: Hallo.', '00:01:23', False),
    ('Wir prüfen morgen früh.', 'morgen früh', False),
    ('Termin am 07.04.1983.', '07.04.1983', False),
    ('Geburtsdatum: 07.04.1983.', '07.04.1983', True),
    ('Sie wurde am 7. April 1983 geboren.', '7. April 1983', True),
    ('Geburtsdatum wird später besprochen. [00:01:23]', '00:01:23', False),
    ('Anna, geb. 07.04.1983.', '07.04.1983', True),
    ('Geburtsdatum: morgen früh.', 'morgen früh', False),
    ('Geburtstag ist nächste Woche.', 'nächste Woche', False),
    ('Geboren am 07.04.1983, Termin am 08.04.2026.', '08.04.2026', False),
    ('Geburtsdatum: 07.04.1983.', 'Geburtsdatum: 07.04.1983', True),
])
def test_birth_date_needs_birth_context_not_just_model_score(text, value, expected, tmp_path):
    detector = Detector(load_policy(None), tmp_path)
    detector._model = Predictions('date_of_birth', value)
    dates = [f for f in detector.detect([{'id': 's1', 'text': text}]) if f['category'] == 'date_of_birth']
    assert bool(dates) is expected


@pytest.mark.parametrize('label,value,text', [
    ('username', 'SAP', 'Wir verwenden SAP.'),
    ('username', 'Account Manager', 'Account Manager: Hallo.'),
    ('person', 'Kunde', 'Der Kunde hat zugestimmt.'),
    ('first_name', '1', '[00:01:23] Sprecher 1: Hallo.'),
    ('person', 'Zwei Personen', 'Zwei Personen prüfen gemeinsam.'),
    ('date_of_birth', '00:01:23', 'Geburtsdatum: [00:01:23]'),
])
def test_nonpersonal_type_evidence_covers_alternate_model_labels(label, value, text, tmp_path):
    detector = Detector(load_policy(None), tmp_path)
    detector._model = Predictions(label, value)
    assert detector.detect([{'id': 's1', 'text': text}]) == []


def test_explicit_login_context_preserves_real_username_even_if_it_matches_product(tmp_path):
    detector = Detector(load_policy(None), tmp_path)
    detector._model = Predictions('username', 'Atlas')
    assert any(f['category'] == 'username' for f in detector.detect([
        {'id': 's1', 'text': 'Mein Benutzername ist Atlas.'},
    ]))


def test_unambiguous_local_name_variants_share_identity_without_name_list():
    texts = ['Anna Beispiel: Hallo.', 'Anna prüft.', 'A. Beispiel antwortet.',
             'Beispiel bestätigt.', 'Anna Beispiel, genannt Anni, übernimmt.', 'Anni sagt zu.']
    findings = Detector(load_policy(None), rules_only=True).detect([
        {'id': f's{i}', 'text': text} for i, text in enumerate(texts)
    ])
    key = next(f['entity_key'] for f in findings if f['text'] == 'Anna Beispiel')
    for value in ['Anna', 'A. Beispiel', 'Beispiel', 'Anni']:
        assert any(f['text'] == value and f.get('entity_key') == key and not f.get('review_only') for f in findings)


def test_underscore_speaker_links_spaced_name_and_short_forms():
    texts = ['Max_Muster   0:03\nHallo.', 'Max Muster prüft. Max antwortet. M. Muster stimmt zu.']
    findings = Detector(load_policy(None), rules_only=True).detect([
        {'id': f's{i}', 'text': text} for i, text in enumerate(texts)
    ])
    key = next(f['entity_key'] for f in findings if f['text'] == 'Max_Muster')
    for value in ['Max Muster', 'Max', 'M. Muster']:
        assert any(f['text'] == value and f.get('entity_key') == key for f in findings)


def test_shared_surname_is_redacted_without_picking_a_person():
    findings = Detector(load_policy(None), rules_only=True).detect([
        {'id': 's1', 'text': 'Ada Beispiel: Hallo.\nBen Beispiel: Hallo.\nHerr Beispiel prüft.'},
    ])
    matches = [f for f in findings if f['text'] == 'Beispiel' and not f.get('review_only')]
    assert matches and all('entity_key' not in f for f in matches)


def test_name_collision_in_benign_phrase_is_preserved_but_honorific_is_person():
    findings = Detector(load_policy(None), rules_only=True).detect([
        {'id': 's1', 'text': 'Anna Winter: Hallo.'},
        {'id': 's2', 'text': 'Im Winter arbeiten wir weiter. Frau Winter prüft.'},
    ])
    matches = [f for f in findings if f['segment_id'] == 's2' and not f.get('review_only')]
    assert len(matches) == 1 and matches[0]['start'] == 36


def test_explicit_alias_can_override_non_person_term_classification(tmp_path):
    policy = load_policy(None)
    policy['aliases'] = [{'entity_key': 'person:synthetic', 'aliases': ['Atlas']}]
    findings = Detector(policy, rules_only=True).detect([{'id': 's1', 'text': 'Das System Atlas.'}])
    assert any(f['text'] == 'Atlas' and f.get('entity_key') == 'person:synthetic' for f in findings)


def test_role_prefix_does_not_become_part_of_a_person_or_alias(tmp_path):
    detector = Detector(load_policy(None), tmp_path)
    detector._model = Predictions('full_name', 'Anna Müller')
    findings = detector.detect([{'id': 's1', 'text': 'Account Manager Anna Müller: Hallo.'}])
    assert any(f['text'] == 'Anna Müller' for f in findings)
    assert not any('Account' in f['text'] or 'Manager' in f['text'] for f in findings)
