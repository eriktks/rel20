import ast
import json
import sys

def analyze_logfile(logfile_name):
    logfile = open(logfile_name, "r")
    lines = logfile.readlines()
    logfile.close()
    entities = []
    doc_lengths = []
    l = 0
    for line in lines:
        l += 1
        line_json = ast.literal_eval(line.strip())
        entities.append([])
        for mention in line_json['API_DOC']:
            entities[-1].append(str(mention["pos"]) + " " + str(mention["end_pos"]) + " " + mention["mention"])
    return entities

doc_id = -1
if len(sys.argv) > 2 and len(sys.argv) < 5:
    logfile_name_1 = sys.argv[1]
    logfile_name_2 = sys.argv[2]
    if len(sys.argv) > 3:
        doc_id = int(sys.argv[3])
else:
    sys.exit("usage: python3 logfile_analyze.py logfile_name_1 logfile_name_2 [doc_id]")

entities_1 = analyze_logfile(logfile_name_1)
entities_2 = analyze_logfile(logfile_name_2)

if doc_id >= 0:
    for j in range(0, len(entities_1[doc_id])):
        if entities_1[doc_id][j] in entities_2[doc_id]:
            print("*", entities_1[doc_id][j])
    print("")
    for j in range(0, len(entities_1[doc_id])):
        if not entities_1[doc_id][j] in entities_2[doc_id]:
            print("1", entities_1[doc_id][j])
    print("")
    for j in range(0, len(entities_2[doc_id])):
        if not entities_2[doc_id][j] in entities_1[doc_id]:
            print("2", entities_2[doc_id][j])
else:
    max_distance = -1
    max_i = -1
    for i in range(0, len(entities_1)):
        found = 0
        try:
            for j in range(0, len(entities_1[i])):
                if entities_1[i][j] in entities_2[i]:
                    found += 1
            distance = len(entities_1[i]) + len(entities_2[i]) - 2 * found
            print(i, distance, len(entities_1[i]), len(entities_2[i]), found)
            if distance > max_distance:
                max_distance = distance
                max_i = i
        except:
            pass
    print(f"largest distance was {max_distance} for document {max_i}")
