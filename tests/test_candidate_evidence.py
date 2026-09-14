"""Synthetic type-evidence tests; no transcript-derived fixtures."""
import re

import pytest

from transcript_anonymizer.detection import Detector, load_policy


class ConfidentModel:
    def __init__(self, category, value):
        self.category, self.value = category, value

    def extract_entities(self, text, labels, **kwargs):
        return {'entities': {self.category: [
            {'text': m.group(), 'start': m.start(), 'end': m.end(), 'score': 0.99}
            for m in re.finditer(re.escape(self.value), text)
        ]}}


def proposals(label, value, text):
    detector = Detector(load_policy(None))
    detector._model = ConfidentModel(label, value)
    return [f for f in detector.detect([{'id': 's1', 'text': text}]) if not f.get('review_only')]


@pytest.mark.parametrize('label,value,text', [
    ('email', 'Nachrichten', 'Wir beantworten Nachrichten.'),
    ('iban', 'Sparkasse', 'Die Sparkasse verwendet ein Portal.'),
    ('account_id', 'Referenznummer', 'Das Formular hat eine Referenznummer.'),
    ('username', 'Okay.\nWeiter', 'Okay.\nWeiter'),
    ('username', 'Anwender', 'Der Anwender öffnet die Maske.'),
    ('username', 'Regional Operations Manager', 'Regional Operations Manager ist eine Rolle.'),
    ('account_id', 'Regional Operations Manager', 'Regional Operations Manager ist eine Rolle.'),
    ('person', 'Bereichsvorstand', 'Der Bereichsvorstand entscheidet.'),
    ('person', 'Leitungsebene', 'Die Leitungsebene prüft das.'),
    ('person', 'Nachname', 'Im Formular fehlt der Nachname.'),
    ('person', 'Erwachsene', 'Erwachsene können teilnehmen.'),
    ('person', 'Automationswerkzeug_7', 'Das Automationswerkzeug_7 startet.'),
    ('username', 'Automationswerkzeug_7', 'Das Automationswerkzeug_7 startet.'),
    ('person', 'Nimbus', 'Das Programm nennt sich Nimbus.'),
    ('person', 'P. Q.', 'Die Prozessvarianten heißen P. Q. und R. S.'),
    ('person', 'Mhm.', 'Mhm.\nMhm.'),
    ('person', 'Gut.', 'Gut.\nGenau.'),
    ('person', 'Genau.', 'Genau.\nJa.'),
    ('person', 'Z', 'Zuerst kommt der Buchstabe Z.'),
    ('person', 'Österreicher', 'Österreicher in Österreich nutzen andere Systeme.'),
    ('address', 'Österreich', 'Die Systeme laufen in Österreich und Frankreich.'),
    ('address', 'Frankreich', 'Sie wohnt in Wien. Die Daten liegen in Frankreich.'),
    ('username', 'Prüfspezialist_7', 'Prüfspezialist_7: Guten Tag.'),
])
def test_type_contradictions_do_not_become_replacements(label, value, text):
    assert not proposals(label, value, text)


@pytest.mark.parametrize('label,value,text', [
    ('username', 'Nimbus', 'Mein Login lautet Nimbus.'),
    ('username', 'a.beispiel_72', 'Der Benutzername lautet a.beispiel_72.'),
    ('account_id', 'ABX-778899', 'Die Kontonummer ist ABX-778899.'),
    ('email', 'eva@example.invalid', 'Kontakt: eva@example.invalid'),
    ('email', 'eva [at] example [dot] invalid', 'Kontakt: eva [at] example [dot] invalid'),
    ('iban', 'DE89370400440532013000', 'IBAN: DE89370400440532013000'),
    ('iban', 'DE00370400440532013000', 'IBAN: DE00370400440532013000'),
    ('person', 'Ada Beispiel', 'Bereichsvorstand Ada Beispiel prüft.'),
    ('person', 'Schweizer', 'Frau Schweizer übernimmt.'),
    ('person', 'Hi', 'Hi erklärt den Ablauf.'),
    ('person', 'P. Q.', 'Herr P. Q. hat angerufen.'),
    ('address', 'Wiesenweg 42', 'Sie wohnt im Wiesenweg 42.'),
    ('address', 'Wien', 'Ihr Wohnort ist Wien.'),
])
def test_identifying_evidence_preserves_real_candidates(label, value, text):
    assert any(f['text'] == value for f in proposals(label, value, text))


def test_rejected_role_cannot_seed_document_aliases():
    detector = Detector(load_policy(None))
    detector._model = ConfidentModel('person', 'Regional Operations Manager')
    findings = detector.detect([
        {'id': 's1', 'text': 'Regional Operations Manager ist eine Rolle.'},
        {'id': 's2', 'text': 'Regional ist der Geltungsbereich.'},
    ])
    assert not [f for f in findings if not f.get('review_only')]


def test_model_false_full_name_cannot_seed_document_aliases():
    detector = Detector(load_policy(None))
    detector._model = ConfidentModel('person', 'Mhm. Mhm.')
    findings = detector.detect([
        {'id': 's1', 'text': 'Mhm. Mhm.'},
        {'id': 's2', 'text': 'Mhm.'},
    ])
    assert not findings


def test_short_name_is_not_a_greeting_inside_speaker_label():
    detector = Detector(load_policy(None), rules_only=True)
    found = detector.detect([{'id': 's1', 'text': 'Minh Hi: Guten Tag.'}])
    assert any(f['text'] == 'Minh Hi' for f in found)


def test_context_before_chunk_boundary_is_not_lost():
    text = 'Das Programm nennt sich Nimbus.'
    offset = text.index('Nimbus')
    detector = Detector(load_policy(None))
    found = detector._parse_model_result('s1', text, offset, text[offset:], {
        'entities': {'person': [{'text': 'Nimbus', 'start': 0, 'end': 6, 'score': 0.99}]}})
    assert not found


def test_weak_handles_and_places_remain_in_ranked_review():
    from transcript_anonymizer.ranking import rank_retained_passages
    for label, value in [('username', 'Aurora'), ('address', 'Florenz')]:
        detector = Detector(load_policy(None))
        detector._model = ConfidentModel(label, value)
        segments = [{'id': 's1', 'text': value}]
        found = detector.detect(segments)
        assert found and all(f.get('review_only') for f in found)
        assert rank_retained_passages(segments, segments, found)
