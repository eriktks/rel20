import argparse
import evaluate_predictions
import json
import numpy as np
import requests
import re
import sys

from REL.training_datasets import TrainingEvaluationDatasets

parser = argparse.ArgumentParser()
parser.add_argument("--max_docs", help = "number of documents")
parser.add_argument("--process_sentences", help = "process sentences rather than documents", action="store_true")
parser.add_argument("--split_docs", help = "split documents")
parser.add_argument("--use_bert", help = "use Bert rather than Flair", action="store_true")
parser.add_argument("--use_bert_base", help = "use Bert base rather than Flair", action="store_true")
parser.add_argument("--use_server", help = "use server", action="store_true")
parser.add_argument("--wiki_version", help = "Wiki version")
args = parser.parse_args()

np.random.seed(seed=42)

base_url = "/store/userdata/etjong/REL.org/data/"
if args.max_docs:
    max_docs = int(args.max_docs)
else:
    max_docs = 50
if args.process_sentences:
    process_sentences = True
else:
    process_sentences = False
if args.split_docs:
    split_docs = True
else:
    split_docs = False
if args.wiki_version:
    wiki_version = args.wiki_version
else:
    wiki_version = "wiki_2019"

datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()["aida_testB"]

if args.use_server:
    use_server = True
else:
    use_server = False
if args.use_bert:
    use_bert = True
else:
    use_bert = False
    use_sentences = True
if args.use_bert_base:
    use_bert = True
    use_bert_base = True
else:
    use_bert_base = False

print(f"max_docs={max_docs} wiki_version={wiki_version} use_bert={use_bert} use_bert_base={use_bert_base} use_server={use_server} process_sentences={process_sentences}")

docs = {}
all_results = {}
for i, doc in enumerate(datasets):
    sentences = []
    for x in datasets[doc]:
        if x["sentence"] not in sentences:
            sentences.append(x["sentence"])
    text = ". ".join([x for x in sentences])

    if len(docs) >= max_docs:
        print(f"length docs is {len(docs)}.")
        print("====================")
        break

    if len(text.split()) > 200:
        docs[doc] = [text, []]
        # Demo script that can be used to query the API.
        if use_server:
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
                    results_list.append({ "mention": result[2], "prediction": result[3] }) # Flair + Bert
                all_results[doc] = results_list
                print (results.json())
            except json.decoder.JSONDecodeError:
                print("The analysis results are not in json format:", str(results))
                all_results[doc] = []

if len(all_results) > 0:
    evaluate_predictions.evaluate(all_results)


# --------------------- Now total --------------------------------
# ------------- RUN SEPARATELY TO BALANCE LOAD--------------------
if not use_server:
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
    if use_bert:
        if use_bert_base:
            tagger_ner = load_bert_ner("dslim/bert-base-NER")
        else:
            tagger_ner = load_bert_ner("dslim/bert-large-NER")
    else:
        tagger_ner = SequenceTagger.load("ner-fast")

    start = time()
    mentions_dataset, n_mentions = mention_detection.find_mentions(docs, use_bert, process_sentences, split_docs, tagger_ner)
    print("MD took: {} seconds".format(round(time() - start, 2)))

    # 3. Load model.
    config = {
        "mode": "eval",
        "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
    }
    model = EntityDisambiguation(base_url, wiki_version, config)

    # 4. Entity disambiguation.
    start = time()
    predictions, timing = model.predict(mentions_dataset)
    print("ED took: {} seconds".format(round(time() - start, 2)))

    #for doc in docs:
    #    print(len(re.sub("[^ ]+","",docs[doc][0])), docs[doc][0])
    evaluate_predictions.evaluate(predictions)
