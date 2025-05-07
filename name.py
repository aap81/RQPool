DATADIR = 'datasets'


ADJ = '_A.txt'
GRAPHIND = '_graph_indicator.txt'
GRAPHLABEL = '_graph_labels.txt'
NODEATTR = '_node_attributes.txt'
NEWLABEL = '_graph_newlabel.txt'

TRAIN = '_train.txt'
VAL = '_val.txt'
TEST = '_test.txt'

NODELABEL = '_node_labels.txt'
NODEATTR = '_node_attributes.txt'
DATASETS = ["COX2", "DHFR", "PROTEINS", "DD", "AIDS", "NCI1", "NCI109", "PC-3", "MCF-7", "MOLT-4", "UACC257", "SN12C", "SF-295", "NCI-H23", "OVCAR-8", "SW-620", "P388"]
INTERGRAPHS = ["sort", "set2set", "sage", "topk", "sageMeanMax", "attention", "sagPool", 'mean', 'none']
TESTING_SETS = DATASETS