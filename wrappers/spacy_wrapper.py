from typing import List

import spacy


class SpacyWrapper:
    def __init__(self, spacy_module: str) -> None:
        try:
            self._nlp = spacy.load(spacy_module)
        except OSError:
            spacy.cli.download(spacy_module)
            self._nlp = spacy.load(spacy_module)

    def sentence_tokenizer(self, text: str) -> List[str]:
        """
        Tokenize (split) text in sentences.

        for example:
            sentence_tokenizer("Hello, world. Here are two sentences.")
        will output:
            ['Hello, world.', 'Here are two sentences.']
        :param text: raw text to split into sentences
        :return: list of strings, each string is a sentence.
        """
        doc = self._nlp(text)
        return [sent.string.strip() for sent in doc.sents]
