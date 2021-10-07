import hdbscan
import numpy as np

from collections import defaultdict

def generate_groups(utterances, embeddings, metric = 'euclidean'):

    keys = ['text', 'intent', 'confidence']
    common_examples = []
    clusterer = hdbscan.HDBSCAN(
        metric=metric,
        min_cluster_size=5,
        min_samples=2,
        prediction_data=True,
        cluster_selection_method='eom',
        alpha=0.8 # TODO: The docs say this should be left alone, and keep the default of 1, but playing with it seems to help, might be different with real data.
        ).fit(np.inner(embeddings, embeddings))

    # create list like: [ [utterance, label] ] with strings
    labels_strings = list(map(str, clusterer.labels_))
    cluster_probs = hdbscan.all_points_membership_vectors(clusterer)
    values = zip(utterances, labels_strings, cluster_probs)
    for value in values:
        common_examples.append(dict(zip(keys, value)))

    message_groups = defaultdict(list)
    for example in common_examples:
        message_groups[example['intent']].append({
            "phrase": str(example['text']),
            # "confidence": list(example['confidence'])
        })

    unlabeled_messages = list(clusterer.labels_).count(-1)
    total_messages = len(utterances)
    return {
        "intents found": int(clusterer.labels_.max()),
        "unlabeled messages": int(unlabeled_messages),
        "labeled messaged": int(total_messages - unlabeled_messages),
        "total messages": int(total_messages),
        "message groups": message_groups
    }