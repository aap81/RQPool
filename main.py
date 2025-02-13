import argparse
import test
from name import *
from utils import log_print
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='AIDS', help='Dataset used')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batchsize', type=int, default=256, help='Training batch size')
parser.add_argument('--nepoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--hdim', type=int, default=64, help='Hidden feature dim')
parser.add_argument('--width', type=int, default=4, help='Width of GCN')
parser.add_argument('--depth', type=int, default=6, help='Depth of GCN')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
parser.add_argument('--normalize', type=int, default=1, help='Whether batch normalize')
parser.add_argument('--beta', type=float, default=0.999, help='CB loss beta')
parser.add_argument('--gamma', type=float, default=1.5, help='CB loss gamma')
parser.add_argument('--decay', type=float, default=0, help='Weight decay')
parser.add_argument('--seed', type=int, default=10, help='Random seed')
parser.add_argument('--patience', type=int, default=50, help='Patience')
parser.add_argument('--intergraph', default='sort', help="mean or max or attention or none")
parser.add_argument('--alltests', type=int, default=0, help='Run all tests for the data and hyperparameter')
parser.add_argument('--datagroup', type=int, default=0, help="select dataset group")
parser.add_argument('--logfile', default='output.txt', help="set output file for logs")
parser.add_argument('--metricsfile', default='metrics-log.txt', help="set output file for metrics results")
parser.add_argument('--completedindex', type=int, default=-1, help="completed_index")
parser.add_argument('--endindex', type=int, default=999999999, help="end_index")
parser.add_argument('--enableprint', type=int, default=0, help="enables print statements")

args = parser.parse_args()

#checking all hyper parameters with intergraph max and mean with integraph 0 (disabled)
if args.alltests == 2:
    datasets = [TESTING_SETS[args.datagroup]]
    index = 0
    total_tests = len(INTERGRAPHS) * len(datasets)
    for dataset in datasets:
        for intergraph in INTERGRAPHS:
            if (index > args.completedindex and index < args.endindex):
                args.data = dataset
                args.intergraph = intergraph
                args.lr = 1e-3
                args.batchsize = 256
                args.hdim = 64
                args.width = 4
                args.depth = 6
                args.dropout = 0.4
                args.decay = 0  # Set decay value
                log_print(f"Group by {dataset}, Test number: {index + 1}/{total_tests}, Intergraph: {args.intergraph}", args.logfile)
                test.execute(args)
            else:
                log_print(f"Group by {dataset}, Test number: {index + 1}/{total_tests} skipped, Intergraph: {args.intergraph}", args.logfile)
            index += 1
else:
    test.execute(args)
