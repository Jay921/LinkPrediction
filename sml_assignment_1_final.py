import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import random
import csv
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json

training_data = open("train.txt", "r")
train_graph = nx.Graph()

def getNodes(training_data):
    nodes = []

    for line in training_data:
        line_split = line.split('\n')
        space_split = line_split[0].split(' ')
        nodes.append(int(space_split[0]))

    return nodes

connected_nodes = getNodes(training_data)

def get_isolated_nodes(connected_nodes):
    isolated_nodes = []
    for i in range(4085):
        if i not in connected_nodes:
            isolated_nodes.append(i)

    return isolated_nodes

isolated_nodes = get_isolated_nodes(connected_nodes)

total_nodes = connected_nodes + isolated_nodes

train_graph.add_nodes_from(total_nodes)

training_data_list = pd.read_csv('train.txt', header=None)

tr_d = list(training_data_list[0])

def getDictionary(tr_d):
    nodes = {}
    for line in tr_d:
        space_split = line.split(" ")
        node_1 = [space_split[0]]
        relations = space_split[1].split("\t")
        node_relations = node_1 + relations
        nodes.update({node_relations[0]: 0})
        nodes[node_relations[0]] = relations

    return nodes

def getDictEdges(n_relations):
    edges = []
    for node in n_relations.keys():
        for destNode in n_relations[node]:
            e = (int(node),int(destNode))
            edges.append(e)

    return edges

node_edges = getDictEdges(getDictionary(tr_d))

train_graph.add_edges_from(node_edges)

positive_edges = train_graph.edges()

# Creating the Dataset------------------------------------------------------------
source_nodes = [edge[0] for edge in positive_edges]
dest_nodes = [edge[1] for edge in positive_edges]

df_pos = pd.DataFrame()
df_pos["Source"] = source_nodes
df_pos["Sink"] = dest_nodes

## Negative edges
def neg_edges():
    non_edges = set([])
    while (len(non_edges)<26937):
        a = random.randint(0, 4084)
        b = random.randint(0, 4084)
        if a!=b and ((a,b) not in positive_edges):
            non_edges.add((a, b))
        else:
            continue
    return non_edges

negative_edges = neg_edges()

source_nodes_negative = [edge[0] for edge in negative_edges]
dest_nodes_negative = [edge[1] for edge in negative_edges]

df_neg = pd.DataFrame()
df_neg["Source"] = source_nodes_negative
df_neg["Sink"] = dest_nodes_negative

## train_test_split
X_train_pos, X_test_pos, y_train_pos, y_test_pos  = train_test_split(df_pos,np.ones(len(df_pos)),test_size=0.04, random_state=9)
X_train_neg, X_test_neg, y_train_neg, y_test_neg  = train_test_split(df_neg,np.zeros(len(df_neg)),test_size=0.04, random_state=9)

X_train_pos["Labels"] = y_train_pos
X_train_neg["Labels"] = y_train_neg

train_df = pd.concat([X_train_pos, X_train_neg], ignore_index=True)


X_test_pos["Labels"] = y_test_pos
X_test_neg["Labels"] = y_test_neg

test_from_train_df = pd.concat([X_test_pos, X_test_neg], ignore_index=True)

print("Number of nodes in the graph with edges", df_pos.shape[0])
print("Number of nodes in the graph without edges", df_neg.shape[0])


# Feature Extraction----------------------------------------------------------------

#node features
with open('nodes.json', 'r') as f:
    nodes_dict = json.load(f)


#keywordSimilarity--------------------------------------------
def keyWord_ (nodes_dict):
    keywords = []
    for i in range(len(nodes_dict)):
        keywords_no = []
        for j in range(0,53):
            if ("keyword_{}".format(j) in nodes_dict[i].keys()) and (i == nodes_dict[i]["id"]):
                keywords_no.append(1)
            else:
                keywords_no.append(0)

        keywords.append(keywords_no)

    return keywords

keyW = np.array(keyWord_(nodes_dict))

def cosine_keyword(sourceNode, destNode):
    sourceNode_r = sourceNode.reshape(1,53)
    destNode_r = destNode.reshape(1,53)
    cos_lib = cosine_similarity(sourceNode_r, destNode_r)
    return cos_lib

SourceList = list(train_df["Source"])
DestinationList = list(train_df["Sink"])

print(len(SourceList))

SourceList_t = list(test_from_train_df["Source"])
DestinationList_t = list(test_from_train_df["Sink"])

def cosine_df_keyword_train(keyW, SourceList, DestinationList):
    cos_values = []
    for i in range(len(train_df)):
        cos_values.append(cosine_keyword(keyW[int(SourceList[i])], keyW[int(DestinationList[i])])[0][0])

    return cos_values

def cosine_df_keyword_dev(keyW, SourceList, DestinationList):
    cos_values = []
    for i in range(len(test_from_train_df)):
        cos_values.append(cosine_keyword(keyW[int(SourceList[i])], keyW[int(DestinationList[i])])[0][0])

    return cos_values

def cosine_df_keyword_test(keyW, SourceList, DestinationList):
    cos_values = []
    for i in range(len(test_df)):
        cos_values.append(cosine_keyword(keyW[int(SourceList[i])], keyW[int(DestinationList[i])])[0][0])

    return cos_values
#keywordSimilarity--------------------------------------------

#Venues Similarity--------------------------------------------
def venue_ (nodes_dict):
    venues = []
    for i in range(len(nodes_dict)):
        venues_no = []
        for j in range(0,348):
            if "venue_{}".format(j) in nodes_dict[i].keys():
                venues_no.append(1)
            else:
                venues_no.append(0)

        venues.append(venues_no)

    return venues

venueW = np.array(venue_(nodes_dict))

print(len(venueW))

def cosine_venue(sourceNode, destNode):
    sourceNode_r = sourceNode.reshape(1,348)
    destNode_r = destNode.reshape(1,348)
    cos_lib = cosine_similarity(sourceNode_r, destNode_r)
    return cos_lib

def cosine_df_venue_train(venueW, SourceList, DestinationList):
    cos_values = []
    for i in range(len(train_df)):
        cos_values.append(cosine_venue(venueW[int(SourceList[i])], venueW[int(DestinationList[i])])[0][0])

    return cos_values

def cosine_df_venue_dev(venueW, SourceList, DestinationList):
    cos_values = []
    for i in range(len(test_from_train_df)):
        cos_values.append(cosine_venue(venueW[int(SourceList[i])], venueW[int(DestinationList[i])])[0][0])

    return cos_values

def cosine_df_venue_test(venueW, SourceList, DestinationList):
    cos_values = []
    for i in range(len(test_df)):
        cos_values.append(cosine_venue(venueW[int(SourceList[i])], venueW[int(DestinationList[i])])[0][0])

    return cos_values

#Venues Similarity-------------------------------------------- 

##Resource Allocation
def get_resource_allocation(edge):
    try:
        return list(nx.resource_allocation_index(train_graph, [edge]))[0][2]
    except:
        return 0
    
##Common Neighbours
def get_common_neighbours(source, destination):
    try:
        return len(sorted(nx.common_neighbors(train_graph, source, destination)))
    except:
        return 0
    
##Jaccard Coefficient
def get_jaccard_coefficient(edge):
    try:
        return list(nx.jaccard_coefficient(train_graph, [edge]))[0][2]
    except:
        return 0
    
##Adamic Adar
def get_adamic_adar(edge):
    try:
        return list(nx.adamic_adar_index(train_graph, [edge]))[0][2]
    except:
        return 0
    
##Preferential Attachment
def get_preferential_attachment(edge):
    try:
        return list(nx.preferential_attachment(train_graph, [edge]))[0][2]
    except:
        return 0
    
##Shortest path
def get_shortest_path_length(a,b):
    p=-1
    try:
        if train_graph.has_edge(a,b):
            train_graph.remove_edge(a,b)
            p= nx.shortest_path_length(train_graph,source=a,target=b)
            train_graph.add_edge(a,b)
        else:
            p= nx.shortest_path_length(train_graph,source=a,target=b)
        return p
    except:
        return -1

##Katz Cenrtality
# adjacency_spectrum = nx.adjacency_spectrum(train_graph)
# max_alpha = round(1.0/max(adjacency_spectrum.real), 3)
max_alpha_r = 0.015
katz = nx.katz_centrality(train_graph, alpha=max_alpha_r, beta=1.0)
mean_katz = float(sum(katz.values())) / len(katz)

##Hits
hits = nx.hits(train_graph, max_iter=100, tol=1e-08, nstart=None, normalized=True)

##PageRank
pageRank = nx.pagerank(train_graph, alpha=0.85)
mean_pr=float(sum(pageRank.values())) / len(pageRank)

# nx.shortest_path_length(train_graph,source=a,target=b)

# Public-test set-----------------------
test_df = pd.read_csv('test-public.csv')

# Generating features for train set
train_df["Keywords"] = cosine_df_keyword_train(keyW, SourceList, DestinationList)
train_df["Venues"] = cosine_df_venue_train(venueW, SourceList, DestinationList)
train_df['ra'] = train_df.apply(lambda row: get_resource_allocation((row['Source'],row['Sink'])),axis=1)
train_df['cn'] = train_df.apply(lambda row: get_common_neighbours(row['Source'],row['Sink']),axis=1)
train_df['jcc'] = train_df.apply(lambda row: get_jaccard_coefficient((row['Source'],row['Sink'])),axis=1)
train_df['adamic'] = train_df.apply(lambda row: get_adamic_adar((row['Source'],row['Sink'])),axis=1)
train_df['pref_a'] = train_df.apply(lambda row: get_preferential_attachment((row['Source'],row['Sink'])),axis=1)
train_df['katz'] = (train_df.Source.apply(lambda x: katz.get(x,mean_katz)) + train_df.Sink.apply(lambda x: katz.get(x,mean_katz))) / 2
train_df['hubs'] = (train_df.Sink.apply(lambda x: hits[0].get(x,0)) + train_df.Sink.apply(lambda x: hits[0].get(x,0))) / 2
train_df['page_rank'] = (train_df.Source.apply(lambda x:pageRank.get(x,mean_pr)) + train_df.Sink.apply(lambda x:pageRank.get(x,mean_pr))) / 2
train_df['Shortest_Path'] = train_df.apply(lambda row: get_shortest_path_length(row['Source'],row['Sink']),axis=1)

# Generating features for dev set
test_from_train_df["Keywords"] = cosine_df_keyword_dev(keyW, SourceList_t, DestinationList_t)
test_from_train_df["Venues"] = cosine_df_venue_dev(venueW, SourceList_t, DestinationList_t)
test_from_train_df['ra'] = test_from_train_df.apply(lambda row: get_resource_allocation((row['Source'],row['Sink'])),axis=1)
test_from_train_df['cn'] = test_from_train_df.apply(lambda row: get_common_neighbours(row['Source'],row['Sink']),axis=1)
test_from_train_df['jcc'] = test_from_train_df.apply(lambda row: get_jaccard_coefficient((row['Source'],row['Sink'])),axis=1)
test_from_train_df['adamic'] = test_from_train_df.apply(lambda row: get_adamic_adar((row['Source'],row['Sink'])),axis=1)
test_from_train_df['pref_a'] = test_from_train_df.apply(lambda row: get_preferential_attachment((row['Source'],row['Sink'])),axis=1)
test_from_train_df['katz'] = (test_from_train_df.Source.apply(lambda x: katz.get(x,mean_katz)) + test_from_train_df.Sink.apply(lambda x: katz.get(x,mean_katz))) / 2
test_from_train_df['hubs'] = (test_from_train_df.Source.apply(lambda x: hits[0].get(x,0)) + test_from_train_df.Sink.apply(lambda x: hits[0].get(x,0))) / 2
test_from_train_df['page_rank'] = (test_from_train_df.Source.apply(lambda x:pageRank.get(x,mean_pr)) + test_from_train_df.Sink.apply(lambda x:pageRank.get(x,mean_pr))) / 2
test_from_train_df['Shortest_Path'] = test_from_train_df.apply(lambda row: get_shortest_path_length(row['Source'],row['Sink']),axis=1)

# Generating features for test set
test_df["Keywords"] = cosine_df_keyword_test(keyW, SourceList, DestinationList)
test_df["Venues"] = cosine_df_venue_test(venueW, SourceList, DestinationList)
test_df['ra'] = test_df.apply(lambda row: get_resource_allocation((row['Source'],row['Sink'])),axis=1)
test_df['cn'] = test_df.apply(lambda row: get_common_neighbours(row['Source'],row['Sink']),axis=1)
test_df['jcc'] = test_df.apply(lambda row: get_jaccard_coefficient((row['Source'],row['Sink'])),axis=1)
test_df['adamic'] = test_df.apply(lambda row: get_adamic_adar((row['Source'],row['Sink'])),axis=1)
test_df['pref_a'] = test_df.apply(lambda row: get_preferential_attachment((row['Source'],row['Sink'])),axis=1)
test_df['katz'] = (test_df.Source.apply(lambda x: katz.get(x,mean_katz)) + test_df.Sink.apply(lambda x: katz.get(x,mean_katz))) / 2
test_df['hubs'] = (test_df.Sink.apply(lambda x: hits[0].get(x,0)) + test_df.Sink.apply(lambda x: hits[0].get(x,0))) / 2
test_df['page_rank'] = (test_df.Source.apply(lambda x:pageRank.get(x,mean_pr)) + test_df.Sink.apply(lambda x:pageRank.get(x,mean_pr))) / 2
test_df['Shortest_Path'] = test_df.apply(lambda row: get_shortest_path_length(row['Source'],row['Sink']),axis=1)



print(nx.info(train_graph))
print(train_df)
print(test_from_train_df)
print(test_df)

train_df.to_csv('train_df.csv')
test_from_train_df.to_csv('dev_df.csv')
test_df.to_csv('test_df.csv')

