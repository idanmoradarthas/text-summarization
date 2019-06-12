import pandas

from wrappers.tensorflow_wrapper import TensorflowWrapper


def test_append_scores():
    sentence_pairs = pandas.DataFrame([["Universal Sentence Encoder embeddings also support short paragraphs.",
                                        "There is no hard limit on how long the paragraph is."],
                                       ["Universal Sentence Encoder embeddings also support short paragraphs.",
                                        "Roughly, the longer the more 'diluted' the embedding will be."],
                                       ["There is no hard limit on how long the paragraph is.",
                                        "Roughly, the longer the more 'diluted' the embedding will be."]],
                                      columns=["sent_1", "sent_2"])

    tensorflow_wrapper = TensorflowWrapper("https://tfhub.dev/google/universal-sentence-encoder-large/3")

    tensorflow_wrapper.append_scores(sentence_pairs)
    tensorflow_wrapper.close()

    result = sentence_pairs["score"].tolist()
    expected = [-0.006272673606872559, -0.19222736358642578, -0.224684476852417]
    assert abs(expected[0] - result[0]) < 0.000001
    assert abs(expected[1] - result[1]) < 0.000001
    assert abs(expected[2] - result[2]) < 0.000001
