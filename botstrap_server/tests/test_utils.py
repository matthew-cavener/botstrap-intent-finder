import json

import tensorflow as tf
import tensorflow_hub as hub

from botstrap_server import utils

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embedding_fn = hub.load(module_url)

def test_groups():

    with open('/usr/src/app/botstrap_server/tests/utterances.json') as f:
        utterances = json.load(f)['utterances']

    embeddings = embedding_fn(utterances)
    groups = utils.generate_groups(utterances, embeddings)
    assert int(groups['intents found']) == 86
