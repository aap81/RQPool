import argparse
import test
from name import *
from utils import log_print

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='AIDS', help='Dataset used')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batchsize', type=int, default=256, help='Training batch size')
parser.add_argument('--nepoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--hdim', type=int, default=128, help='Hidden feature dim')
parser.add_argument('--width', type=int, default=4, help='Width of GCN')
parser.add_argument('--depth', type=int, default=6, help='Depth of GCN')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
parser.add_argument('--normalize', type=int, default=1, help='Whether batch normalize')
parser.add_argument('--beta', type=float, default=0.999, help='CB loss beta')
parser.add_argument('--gamma', type=float, default=1.5, help='CB loss gamma')
parser.add_argument('--decay', type=float, default=0, help='Weight decay')
parser.add_argument('--seed', type=int, default=10, help='Random seed')
parser.add_argument('--patience', type=int, default=50, help='Patience')
parser.add_argument('--intergraph', default='sage', help="mean or max or attention or none")
parser.add_argument('--alltests', type=int, default=0, help='Run all tests for the data and hyperparameter')
parser.add_argument('--datagroup', type=int, default=0, help="select dataset group")
parser.add_argument('--outputfile', default='', help="set output file for logs")
parser.add_argument('--completedindex', type=int, default=0, help="completed_index")
parser.add_argument('--endindex', type=int, default=999999999, help="end_index")

args = parser.parse_args()

if args.outputfile == '':
    args.outputfile = 'output.txt'
else:
    args.outputfile = f"output-{args.outputfile}.txt"

if args.outputfile == '':
    args.outputfile = 'output.txt'
else:
    args.outputfile = f"output-{args.outputfile}.txt"

#checking all hyper parameters with intergraph max and mean with integraph 0 (disabled)
if args.alltests == 1:
    datasets = []
    match args.datagroup:
        case 1:
            completed_index = -1
            end_index = 9999
            datasets = group1
        case 2:
            completed_index = -1
            end_index = 9999
            datasets = group2
        case 3:
            completed_index = -1
            end_index = 9999
            datasets = group3
        case 4:
            completed_index = -1
            end_index = 9999
            datasets = group4
        case 5:
            completed_index = -1
            end_index = 9999
            datasets = group5
        case 6:
            completed_index = -1
            end_index = 9999
            datasets = group6
        case 7:
            completed_index = -1
            end_index = 9999
            datasets = group7
        case 8:
            completed_index = -1
            end_index = 9999
            datasets = group8
        case 9:
            completed_index = -1
            end_index = 9999
            datasets = group9
        case 10:
            completed_index = -1
            end_index = 9999
            datasets = group10
    index = 0
    total_tests = len(datasets)
    for dataset in datasets:
        print(f"Group by {dataset}")
        if index > completed_index and index < end_index:
            args.data = dataset
            args.nepoch = 100
            args.lr = 1e-3
            args.batchsize = 256
            args.hdim = 64
            args.width = 4
            args.depth = 6
            args.dropout = 0.4
            args.decay = 0  # Set decay value
            log_print(f"Test number: {index}/{total_tests}", args.outputfile)
            test.execute(args)
        else:
            log_print(f"Test number: {index}/{total_tests} skipped", args.outputfile)
        index += 1
elif args.alltests == 2:
    datasets = DATASETS
    intergraphs = []
    if args.datagroup == 1:
        intergraphs = ["none", "sage", "mean"]
    else:
        intergraphs = ["sort", "set2set", "max"]

    completed_index = args.completedindex
    end_index = args.endindex
    index = 1
    total_tests = len(datasets) * len(intergraphs)
    
    for dataset in datasets:
        for intergraph in intergraphs:
            args.outputfile = f"output-{intergraph}.txt"
            if index > completed_index and index < end_index:
                args.data = dataset
                args.lr = 1e-3
                args.batchsize = 256
                args.hdim = 64
                args.width = 4
                args.depth = 6
                args.intergraph = intergraph
                args.dropout = 0.4
                args.decay = 0  # Set decay value
                log_print(f"Group by {dataset}, Test number: {index}/{total_tests}", args.outputfile)
                test.execute(args)
            else:
                log_print(f"Test number: {index}/{total_tests} skipped", args.outputfile)
            index += 1
else:
    test.execute(args)
