import numpy as np
import requests
import evaluate_predictions
import json

from REL.training_datasets import TrainingEvaluationDatasets

np.random.seed(seed=42)

base_url = "/store/userdata/etjong/REL.org/data/"
wiki_version = "wiki_2019"
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()["aida_testB"]

# random_docs = np.random.choice(list(datasets.keys()), 50)

server = True
docs = {}
all_results = {}
for i, doc in enumerate(datasets):
    sentences = []
    for x in datasets[doc]:
        if x["sentence"] not in sentences:
            sentences.append(x["sentence"])
    text = ". ".join([x for x in sentences])

    if len(docs) == 50:  # was 50
        print(f"length docs is {len(docs)}.")
        print("====================")
        break

    if len(text.split()) > 200:
        docs[doc] = [text, []]
        # Demo script that can be used to query the API.
        if server:
            myjson = {
                "text": text,
                "spans": [
                    # {"start": 41, "length": 16}
                ],
            }
            print("----------------------------")
            print(i, "Input API:")
            print(myjson)

            print("Output API:")
            results = requests.post("http://0.0.0.0:5555", json=myjson)
            print("----------------------------")
            try:
                results_list = []
                for result in results.json():
                    # results_list.append({ "mention": result[2], "prediction": result[3] }) # Flair
                    results_list.append({ "mention": result[3], "prediction": result[2] }) # Bert
                all_results[doc] = results_list
                print (results.json())
            except json.decoder.JSONDecodeError:
                print("The analysis results are not in json format:", str(results))
                all_results[doc] = []

if len(all_results) > 0:
    evaluate_predictions.evaluate(all_results)


# --------------------- Now total --------------------------------
# ------------- RUN SEPARATELY TO BALANCE LOAD--------------------
if not server:
    from time import time

    import flair
    import torch
    from flair.models import SequenceTagger

    from REL.entity_disambiguation import EntityDisambiguation
    from REL.mention_detection import MentionDetection

    from REL.ner.bert_wrapper import load_bert_ner


    flair.device = torch.device("cpu")

    mention_detection = MentionDetection(base_url, wiki_version)

    # Alternatively use Flair NER tagger.
    #tagger_ner = SequenceTagger.load("ner-fast")
    tagger_ner = load_bert_ner("dslim/bert-large-NER")

    start = time()
    mentions_dataset, n_mentions = mention_detection.find_mentions(docs, tagger_ner)
    print("MD took: {}".format(time() - start))

    # 3. Load model.
    config = {
        "mode": "eval",
        "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
    }
    model = EntityDisambiguation(base_url, wiki_version, config)

    # 4. Entity disambiguation.
    start = time()
    predictions, timing = model.predict(mentions_dataset)
    print("ED took: {}".format(time() - start))

    evaluate_predictions.evaluate(predictions)
