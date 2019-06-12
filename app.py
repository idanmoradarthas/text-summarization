from pathlib import Path

import yaml
from flask import Flask, request, Response, jsonify

from sentence_handler import sentence_pairing, sentence_rank_with_page_rank, sentence_sorter
from wrappers.spacy_wrapper import SpacyWrapper
from wrappers.tensorflow_wrapper import TensorFlowWrapper

app = Flask(__name__)

with open(Path(__file__).parent.joinpath("properties.yaml"), "r") as f:
    properties = yaml.safe_load(f)

spacy_wrapper = SpacyWrapper(properties["spacy-module"])
tensorflow_wrapper = TensorFlowWrapper(properties["universal-sentence-encoder-model"])


@app.route("/summarize/v1.0", methods=["POST"])
def summarize():
    if request.headers['Content-Type'] != 'application/json':
        return Response(status=400)
    document = request.json["doc"]
    sentences = spacy_wrapper.sentence_tokenizer(document)
    df = sentence_pairing(sentences)
    tensorflow_wrapper.append_scores(df)
    result = sentence_rank_with_page_rank(df)
    answer = {"summarized text": sentence_sorter(result, request.json["summarized sentences length"], sentences)}
    resp = jsonify(answer)
    resp.status_code = 200
    return resp


if __name__ == '__main__':
    app.run(port=8080, host="0.0.0.0")
