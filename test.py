import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--loss', type = str)
parser.add_argument('--lr', type = float)
parser.add_argument('--seed', type = int)

args = parser.parse_args()

print('loss', args.loss)
print('lr', args.lr)
print('seed', args.seed)
