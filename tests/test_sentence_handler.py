import pandas

from sentence_handler import sentence_pairing, sentence_rank_with_page_rank, sentence_sorter


def test_sentence_paring():
    sentences = ["Universal Sentence Encoder embeddings also support short paragraphs.",
                 "There is no hard limit on how long the paragraph is.",
                 "Roughly, the longer the more 'diluted' the embedding will be."]
    df = sentence_pairing(sentences)
    expected = pandas.DataFrame([["Universal Sentence Encoder embeddings also support short paragraphs.",
                                  "There is no hard limit on how long the paragraph is."],
                                 ["Universal Sentence Encoder embeddings also support short paragraphs.",
                                  "Roughly, the longer the more 'diluted' the embedding will be."],
                                 ["There is no hard limit on how long the paragraph is.",
                                  "Roughly, the longer the more 'diluted' the embedding will be."]],
                                columns=["sent_1", "sent_2"])
    assert expected.equals(df)


def test_sentence_rank():
    sentence_pairs = pandas.DataFrame([["Universal Sentence Encoder embeddings also support short paragraphs.",
                                        "There is no hard limit on how long the paragraph is."],
                                       ["Universal Sentence Encoder embeddings also support short paragraphs.",
                                        "Roughly, the longer the more 'diluted' the embedding will be."],
                                       ["There is no hard limit on how long the paragraph is.",
                                        "Roughly, the longer the more 'diluted' the embedding will be."]],
                                      columns=["sent_1", "sent_2"])
    sentence_pairs["score"] = [-0.006272673606872559, -0.19222736358642578, -0.224684476852417]
    result = sentence_rank_with_page_rank(sentence_pairs)
    expected = pandas.DataFrame([["Roughly, the longer the more 'diluted' the embedding will be.", 0.47949211540967107],
                                 ["There is no hard limit on how long the paragraph is.", 0.27621153491252437],
                                 ["Universal Sentence Encoder embeddings also support short paragraphs.",
                                  0.24429634967780445]], columns=["sentence", "rank"])
    assert expected.equals(result)


def test_sentence_sorter():
    df = pandas.DataFrame([["Roughly, the longer the more 'diluted' the embedding will be.", 0.47949211540967107],
                           ["There is no hard limit on how long the paragraph is.", 0.27621153491252437],
                           ["Universal Sentence Encoder embeddings also support short paragraphs.",
                            0.24429634967780445]], columns=["sentence", "rank"])
    sentences = ["Universal Sentence Encoder embeddings also support short paragraphs.",
                 "There is no hard limit on how long the paragraph is.",
                 "Roughly, the longer the more 'diluted' the embedding will be."]
    result = sentence_sorter(df, 2, sentences)
    expected = "There is no hard limit on how long the paragraph is. " \
               "Roughly, the longer the more 'diluted' the embedding will be."
    assert expected == result
