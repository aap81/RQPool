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
DATASETS = ["AIDS", "DD", "PROTEINS", 'COX2', 
			"MOLT-4", "SW-620", "NCI-H23", "PC-3", "MCF-7", 
			"NCI109",  "NCI1", "SF-295", "OVCAR-8", "P388",  
			"SN12C", "UACC257", 'DHFR']
INTERGRAPHS = ["sort", "set2set", "sage", "topk", "sageMeanMax", "attention", "sagPool", 'mean', 'none']

TESTING_SETS = ["COX2", "DHFR", "NC1", "NCI109", "PC-3", "MCF-7"]