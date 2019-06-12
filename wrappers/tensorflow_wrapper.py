import pandas
import tensorflow
import tensorflow_hub


class TensorFlowWrapper:
    """
    Wrapper object for TensorFlow graph and helps use it.
    """
    def __init__(self, embedding_layer_hub_name: str) -> None:
        g = tensorflow.Graph()
        with g.as_default():
            # Import the Universal Sentence Encoder's TF Hub module
            embedding_layer = tensorflow_hub.Module(embedding_layer_hub_name)

            self._sts_input1 = tensorflow.placeholder(tensorflow.string, shape=None)
            self._sts_input2 = tensorflow.placeholder(tensorflow.string, shape=None)

            # For evaluation we use exactly normalized rather than approximately normalized.
            self._sts_encode1 = tensorflow.nn.l2_normalize(embedding_layer(self._sts_input1), axis=1)
            self._sts_encode2 = tensorflow.nn.l2_normalize(embedding_layer(self._sts_input2), axis=1)
            cosine_similarities = tensorflow.reduce_sum(tensorflow.multiply(self._sts_encode1, self._sts_encode2),
                                                        axis=1)
            clip_cosine_similarities = tensorflow.clip_by_value(cosine_similarities, -1.0, 1.0)
            self._sim_scores = 1.0 - tensorflow.acos(clip_cosine_similarities)
            init_op = tensorflow.group([tensorflow.global_variables_initializer(), tensorflow.tables_initializer()])
        g.finalize()

        self._session = tensorflow.Session(graph=g)
        self._session.run(init_op)

    def append_scores(self, sentence_pairs: pandas.DataFrame) -> None:
        """
        Appending scoring of cosine similarity based on the given embedding layer.

        :param sentence_pairs: DataFrame matrix of paired sentences with the columns ["sent_1", "sent_2"] where each row
            is a paired sentences.
        :return: None; it append to given DataFrame new column "score" with the cosine similarity score for each pair in
            each row.
        """

        text_a = sentence_pairs["sent_1"].fillna("").tolist()
        text_b = sentence_pairs["sent_2"].fillna("").tolist()

        _, _, scores = self._session.run([self._sts_encode1, self._sts_encode2, self._sim_scores],
                                         feed_dict={self._sts_input1: text_a, self._sts_input2: text_b})

        sentence_pairs["score"] = scores

    def close(self):
        """
        closes the TensorFlow session
        """
        self._session.close()
