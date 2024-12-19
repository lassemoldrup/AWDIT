import linecache
import sys
import time

from mono_txn_graph import MonosatTxnGraph

sys.setrecursionlimit(1000000)


# Causal checking
def run_monosat_txn_graph_causal(file):
    construction_start = time.time()
    raw_ops = linecache.getlines(file)
    index = 0
    for i in range(len(raw_ops)):
        if raw_ops[index] == '\n':
            index = i
            break
    if index != 0:
        raw_ops = raw_ops[0:index]
    causal_hist = MonosatTxnGraph(raw_ops)
    construction_end = time.time()
    construction_time = int((construction_end - construction_start) * 1000)
    print(f"construction: {construction_time} ms")

    wr = causal_hist.get_wr()
    causal_hist.vis_includes(wr)
    causal_hist.vis_is_trans()
    causal_hist.casual_ww()

    encoding_end = time.time()
    encoding_time = int((encoding_end - construction_end) * 1000)
    print(f"encoding: {encoding_time} ms")

    # causal_hist.vis_is_trans()
    result = causal_hist.check_cycle()
    solving_end = time.time()
    solving_time = int((solving_end - encoding_end) * 1000)
    print(f"solving: {solving_time} ms")
    if result:
        print('Find Violation!')
    return result


# Read atomic checking
def run_monosat_txn_graph_ra(file):
    raw_ops = linecache.getlines(file)
    index = 0
    for i in range(len(raw_ops)):
        if raw_ops[i] == '\n':
            index = i
            break
    if index != 0:
        raw_ops = raw_ops[0:index]
    causal_hist = MonosatTxnGraph(raw_ops)
    wr = causal_hist.get_wr()
    causal_hist.vis_includes(wr)
    causal_hist.casual_ww()
    if causal_hist.check_cycle():
        print('Find Violation!')


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_mono_txn.py <hist_path>")
        sys.exit(1)

    if run_monosat_txn_graph_causal(sys.argv[1]):
        print('REJECT')
    else:
        print('ACCEPT')


if __name__ == "__main__":
    main()
