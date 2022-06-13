import numpy as np
import requests
import re

from REL.training_datasets import TrainingEvaluationDatasets

np.random.seed(seed=42)

base_url = "/Users/vanhulsm/Desktop/projects/data/"
base_url = "/home/erikt/projects/rel20/REL/data/"
wiki_version = "wiki_2014"
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()["aida_testB"]

# random_docs = np.random.choice(list(datasets.keys()), 50)

def get_gold_data(doc):
    GOLD_DATA_FILE = "./data/generic/test_datasets/AIDA/AIDA-YAGO2-dataset.tsv"
    entities = []

    in_file = open(GOLD_DATA_FILE, "r")
    for line in in_file:
        if re.search(f"^-DOCSTART- \({doc} ", line):
            break
    for line in in_file:
        if re.search(f"^-DOCSTART- ", line):
            break
        fields = line.strip().split("\t")
        if len(fields) > 3:
            if fields[1] == "B":
                entities.append([fields[2], fields[3]])
    return entities


def compare_elements_with_offset(gold_entities, predicted_entities, predicted_links, gold_i, predicted_i, offset):
    if offset == 0:
        return(set(predicted_links[predicted_i + offset:]) == {-1} and 
               gold_entities[gold_i][0].lower() == predicted_entities[predicted_i][0].lower())
    elif offset < 0:
        return(predicted_i > -offset - 1 and
               set(predicted_links[predicted_i + offset:]) == {-1} and
               gold_entities[gold_i][0].lower() == predicted_entities[predicted_i + offset][0].lower())
    elif offset > 0:
        return(predicted_i < len(predicted_entities) - offset and
               set(predicted_links[predicted_i + offset:]) == {-1} and
               gold_entities[gold_i][0].lower() == predicted_entities[predicted_i + offset][0].lower())
    else:
        print("compare_elements_with_offset: cannot happen")


def compare(gold_entities, predicted_entities):
    gold_links = len(gold_entities) * [-1]
    predicted_links = len(predicted_entities) * [-1]
    for gold_i in range(0, len(gold_entities)):
        predicted_i = int(gold_i * len(predicted_entities) / len(gold_entities))
        for offset in [0, -1, 1, -2, 2, -3, 3]:
            if compare_elements_with_offset(gold_entities, predicted_entities, predicted_links, gold_i, predicted_i, offset):
                gold_links[gold_i] = predicted_i + offset
                predicted_links[predicted_i + offset] = gold_i
                break

    correct = 0
    wrong_md = 0
    wrong_ed = 0
    missed = 0
    for predicted_i in range(0, len(predicted_links)):
        if predicted_links[predicted_i] < 0:
            wrong_md += 1
        elif predicted_entities[predicted_i][1] == gold_entities[predicted_links[predicted_i]][1]:
            correct += 1
        else:
            wrong_ed += 1
    for gold_i in range(0, len(gold_links)):
        if gold_links[gold_i] < 0:
            missed += 1
    return correct, wrong_md, wrong_ed, missed


def evaluate(predictions):
    correct_all = 0
    wrong_md_all = 0
    wrong_ed_all = 0
    missed_all = 0
    for doc in predictions:
        gold_entities = get_gold_data(doc)
        predicted_entities = []
        for mention in predictions[doc]:
            predicted_entities.append([mention["mention"], mention["prediction"]])
        correct, wrong_md, wrong_ed, missed = compare(gold_entities, predicted_entities)
        correct_all += correct
        wrong_md_all += wrong_md
        wrong_ed_all += wrong_ed
        missed_all += missed
    print("Results: PMD RMD PED RED: ", end="")
    print(f"{100*(correct_all+wrong_ed_all)/(correct_all+wrong_ed_all+wrong_md_all):0.1f}% | ",end="")
    print(f"{100*(correct_all+wrong_ed_all)/(correct_all+wrong_ed_all+missed_all):0.1f}% | ", end="")
    print(f"{100*(correct_all)/(correct_all+wrong_ed_all):0.1f}% | ",end="")
    print(f"{100*(correct_all)/(correct_all+missed_all):0.1f}% |")


server = False
number_of_documents = 50
docs = {}
for i, doc in enumerate(datasets):
    sentences = []
    for x in datasets[doc]:
        if x["sentence"] not in sentences:
            sentences.append(x["sentence"])
    text = ". ".join([x for x in sentences])

    if len(docs) == number_of_documents:
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
            print(requests.post("https://rel.cs.ru.nl/api", json=myjson).json())
            print("----------------------------")


# --------------------- Now total --------------------------------
# ------------- RUN SEPARATELY TO BALANCE LOAD--------------------
if not server:
    from time import time

    #import flair
    #import torch
    #from flair.models import SequenceTagger

    from REL.entity_disambiguation import EntityDisambiguation
    from REL.mention_detection import MentionDetection
    from REL.ner.bert_wrapper import load_bert_ner

    #base_url = "C:/Users/mickv/desktop/data_back/"
    base_url = "/home/erikt/projects/rel20/REL/data/"

    # flair.device = torch.device("cuda:0")
    #flair.device = torch.device("cpu")

    mention_detection = MentionDetection(base_url, wiki_version)

    # Alternatively use Flair NER tagger.
    # tagger_ner = SequenceTagger.load("ner-fast")
    tagger_ner = load_bert_ner("dslim/bert-base-NER")

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

    evaluate(predictions)
