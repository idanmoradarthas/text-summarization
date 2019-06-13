from typing import List

import networkx
import numpy
import pandas
from networkx import PowerIterationFailedConvergence
from sklearn.preprocessing import LabelEncoder


def sentence_pairing(sentences: List[str]) -> pandas.DataFrame:
    """
    Create a matrix of paired sentences, where same sentences are omitted.

    :param sentences: list of sentences
    :return: DataFrame with the columns ["sent_1", "sent_2"] where each row is a paired sentences.
    """
    sent_pairs = []
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            if sentences[i] == sentences[j]:
                continue
            sent_pairs.append([sentences[i], sentences[j]])
    return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2"])


def sentence_rank_with_page_rank(sentence_pairs_with_score: pandas.DataFrame) -> pandas.DataFrame:
    """
    Rank the sentences based on their similarity score to each other using page-rank algorithm and output their
    new scores.

    :param sentence_pairs_with_score: DataFrame with the columns ["sent_1", "sent_2", "score"] where each row is a paired sentences with their initial similarity score.
    :return: DataFrame with the columns ["sentence", "rank"] where each sentence has its rank.
    """
    sentences = set()
    sentences.update(sentence_pairs_with_score["sent_1"].tolist())
    sentences.update(sentence_pairs_with_score["sent_2"].tolist())
    sentences_list = list()
    sentences_list.extend(sentences)
    le = LabelEncoder()
    le.fit(sentences_list)
    similarity_matrix = numpy.zeros((len(sentences_list), len(sentences_list)))
    for idx1 in range(len(sentences_list)):
        for idx2 in range(len(sentences_list)):
            if idx1 == idx2:
                # ignore if both are same sentences
                continue
            first_sent = sentences_list[idx1]
            second_sent = sentences_list[idx2]
            df = sentence_pairs_with_score[
                (sentence_pairs_with_score["sent_1"] == first_sent) & (
                        sentence_pairs_with_score["sent_2"] == second_sent)]
            if df.shape[0] == 0:
                df = sentence_pairs_with_score[
                    (sentence_pairs_with_score["sent_1"] == second_sent) & (sentence_pairs_with_score[
                                                                                "sent_2"] == first_sent)]

            similarity_matrix[le.transform([first_sent])[0]][le.transform([second_sent])[0]] = df["score"].iloc[0]
    sentence_similarity_graph = networkx.from_numpy_array(similarity_matrix)
    try:
        scores = networkx.pagerank(sentence_similarity_graph, max_iter=10_000)
    except PowerIterationFailedConvergence:
        scores = networkx.pagerank(sentence_similarity_graph, tol=1)
    result = pandas.DataFrame()
    result["sentence"] = scores.keys()
    result["sentence"] = le.inverse_transform(result["sentence"])
    result["rank"] = scores.values()
    return result


def sentence_sorter(df: pandas.DataFrame, top_n: int, sentences: List[str]) -> str:
    """
    Sort the sentences based on their rank and return the full summarized text in which the sentences appear as they are
    in the given text.

    :param df: DataFrame with the columns ["sentence", "rank"] where each sentence has its rank.
    :param top_n: total number of sentences in the summarized text.
    :param sentences: the given text tokenized into sentences.
    :return: the summarized text
    """
    sorted_df = df.sort_values(by=["rank"], ascending=False)
    selected_sentences = sorted_df.head(top_n)["sentence"].tolist()
    result = [sentence for sentence in sentences if sentence in selected_sentences]
    return " ".join(result)
