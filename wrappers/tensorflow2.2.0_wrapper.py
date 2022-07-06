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

            self._sts_input1 = tensorflow.compat.v1.placeholder(tensorflow.string, shape=None)
            self._sts_input2 = tensorflow.compat.v1.placeholder(tensorflow.string, shape=None)

            # For evaluation we use exactly normalized rather than approximately normalized.
            sts_encode1 = tensorflow.math.l2_normalize(embedding_layer(self._sts_input1), axis=1)
            sts_encode2 = tensorflow.math.l2_normalize(embedding_layer(self._sts_input2), axis=1)
            cosine_similarities = tensorflow.math.reduce_sum(tensorflow.multiply(sts_encode1, sts_encode2),
                                                        axis=1)
            clip_cosine_similarities = tensorflow.clip_by_value(cosine_similarities, -1.0, 1.0)
            self._sim_scores = 1.0 - tensorflow.math.acos(clip_cosine_similarities)
            init_op = tensorflow.group([tensorflow.compat.v1.global_variables_initializer(), tensorflow.compat.v1.tables_initializer()])
        g.finalize()

        self._session = tensorflow.compat.v1.Session(graph=g)
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

        scores = self._session.run(self._sim_scores, feed_dict={self._sts_input1: text_a, self._sts_input2: text_b})

        sentence_pairs["score"] = scores

    def close(self):
        """
        closes the TensorFlow session.
        """
        self._session.close()
