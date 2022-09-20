import re


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


def no_linked_entities_in_rest(predicted_links, predicted_i, offset):
    return(set(predicted_links[predicted_i + offset:]) == {-1})


def compare_elements_with_offset(gold_entities, predicted_entities, predicted_links, gold_i, predicted_i, offset):
    return(no_linked_entities_in_rest(predicted_links, predicted_i, offset) and 
           gold_entities[gold_i][0].lower() == predicted_entities[predicted_i + offset][0].lower())


def compare_md(gold_entities, predicted_entities):
    gold_links = len(gold_entities) * [-1]
    predicted_links = len(predicted_entities) * [-1]
    for gold_i in range(0, len(gold_entities)):
        predicted_i = int(gold_i * len(predicted_entities) / len(gold_entities))
        found = False
        for offset in [0, -1, 1, -2, 2, -3, 3]:
            predicted_i_with_offset = predicted_i + offset
            if (predicted_i_with_offset >= 0 and predicted_i_with_offset < len(predicted_links) and
                predicted_links[predicted_i_with_offset] < 0 and
                compare_elements_with_offset(gold_entities, predicted_entities, predicted_links, gold_i, predicted_i, offset)):
                gold_links[gold_i] = predicted_i + offset
                predicted_links[predicted_i + offset] = gold_i
                found = True
                print("CORRECT", gold_entities[gold_i])
                break
        if not found:
            print("MISSED", gold_entities[gold_i])
    return gold_links, predicted_links


def compare_ed(gold_entities, predicted_entities, gold_links, predicted_links):
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


def compare(gold_entities, predicted_entities):
    gold_links, predicted_links = compare_md(gold_entities, predicted_entities)
    return compare_ed(gold_entities, predicted_entities, gold_links, predicted_links)


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
        print(correct, wrong_md, wrong_ed, missed)
        print(gold_entities)
        print(predicted_entities)
        correct_all += correct
        wrong_md_all += wrong_md
        wrong_ed_all += wrong_ed
        missed_all += missed
    print("Results: PMD RMD PED RED: ", end="")
    print(f"{100*(correct_all+wrong_ed_all)/(correct_all+wrong_ed_all+wrong_md_all):0.1f}% | ",end="")
    print(f"{100*(correct_all+wrong_ed_all)/(correct_all+wrong_ed_all+missed_all):0.1f}% | ", end="")
    print(f"{100*(correct_all)/(correct_all+wrong_ed_all):0.1f}% | ",end="")
    print(f"{100*(correct_all)/(correct_all+missed_all):0.1f}% |")
