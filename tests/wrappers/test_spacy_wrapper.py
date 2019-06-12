from wrappers.spacy_wrapper import SpacyWrapper


def test_sentence_tokenizer():
    spacy_wrapper = SpacyWrapper("en_core_web_sm")
    raw_text = 'Hello, world. Here are two sentences.'
    sentences = spacy_wrapper.sentence_tokenizer(raw_text)
    assert [u'Hello, world.', u'Here are two sentences.'] == sentences
