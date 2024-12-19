import linecache
import sys
import time

from clorm import monkey

monkey.patch()  # must call this before importing clingo
from clorm.clingo import Control
from clorm.orm.core import *
from clorm.orm.factbase import FactBase


class Operation:
    def __init__(self, data_type, var, val, client_id, txn_id):
        self.data_type = data_type
        self.var = var
        self.val = val
        self.client_id = client_id
        self.txn_id = txn_id


class Wtxn(Predicate):
    op_1 = StringField
    op_2 = StringField


class So(Predicate):
    op_1 = StringField
    op_2 = StringField


class Wsv(Predicate):
    op_1 = StringField
    op_2 = StringField


class Wr(Predicate):
    op_1 = StringField
    op_2 = StringField
    op_3 = StringField


class Cc(Predicate):
    op_1 = StringField
    op_2 = StringField


class Co(Predicate):
    op_1 = StringField
    op_2 = StringField


class Tcc(Predicate):
    op_1 = StringField
    op_2 = StringField


class Bad_CyclicSOWR(Predicate):
    op_1 = StringField


class Bad_CyclicCO(Predicate):
    op_1 = StringField


class Show(Predicate):
    op_1 = StringField
    op_2 = StringField


class DiGraph:
    def __init__(self):
        self.adj_map = {}  # node: set(node)

    def add_edge(self, from_node, to_node):
        if from_node in self.adj_map:
            self.adj_map[from_node].add(to_node)
        else:
            self.adj_map[from_node] = {to_node}

    def add_edges(self, from_node, to_nodes):
        if from_node not in self.adj_map:
            self.adj_map[from_node] = set()
        for to_node in to_nodes:
            self.adj_map[from_node].add(to_node)

    def add_vertex(self, new_node):
        if new_node not in self.adj_map:
            self.adj_map[new_node] = set()

    def has_edge(self, from_node, to_node):
        if from_node in self.adj_map and to_node in self.adj_map[from_node]:
            return True
        else:
            return False


class ClingoTxnHistory:
    def __init__(self, ops):
        self.wr_rel = {}
        self.txns = {}
        client_in_so = {}
        r_nodes = {}
        current_tra = []
        self.facts = []
        for i in range(len(ops)):
            op_dict = self.get_op(ops[i])
            if i == len(ops) - 1 or self.get_op(ops[i + 1])['tra_id'] != op_dict['tra_id']:
                if op_dict['client_id'] in client_in_so:
                    self.facts.append(
                        'so("' + str(client_in_so[op_dict['client_id']]) + '","' + str(op_dict['tra_id']) + '")\n')
                client_in_so[op_dict['client_id']] = op_dict['tra_id']
                current_tra.append(op_dict)
                for op in current_tra:
                    if op['op_type'] == 'w':
                        wtxn_str = 'wtxn("' + str(op_dict['tra_id']) + '","' + str(op['var']) + '")\n'
                        if wtxn_str not in self.facts:
                            self.facts.append('wtxn("' + str(op_dict['tra_id']) + '","' + str(op['var']) + '")\n')
                        if op['var'] in self.wr_rel:
                            for key in list(self.wr_rel[op['var']].adj_map):
                                if key == op_dict['tra_id']:
                                    continue
                                wsv_str_1 = 'wsv("' + str(key) + '","' + str(op_dict['tra_id']) + '")\n'
                                wsv_str_2 = 'wsv("' + str(op_dict['tra_id']) + '","' + str(key) + '")\n'
                                if wsv_str_1 not in self.facts:
                                    self.facts.append(wsv_str_1)
                                if wsv_str_2 not in self.facts:
                                    self.facts.append(wsv_str_2)
                            self.wr_rel[op['var']].add_vertex(op_dict['tra_id'])
                        else:
                            graph = DiGraph()
                            graph.add_vertex(op_dict['tra_id'])
                            self.wr_rel[op['var']] = graph
                        if op['var'] in r_nodes:
                            for key in r_nodes[op['var']]:
                                if key != op_dict['tra_id']:
                                    for node in self.txns[key]:
                                        if node['val'] == op['val'] and node['var'] == op['var'] and node[
                                            'op_type'] == 'r':
                                            wr_str = 'wr("' + str(op_dict['tra_id']) + '","' + str(key) + '","' + str(
                                                op['var']) + '")\n'
                                            if wr_str not in self.facts:
                                                self.facts.append(wr_str)
                                            self.wr_rel[op['var']].add_edge(op_dict['tra_id'], key)
                                            break
                    else:
                        if op['var'] in self.wr_rel:
                            has_wr = False
                            for key, t_set in self.wr_rel[op['var']].adj_map.items():
                                if key != op_dict['tra_id']:
                                    for node in self.txns[key]:
                                        if node['val'] == op['val'] and node['var'] == op['var'] and node[
                                            'op_type'] == 'w':
                                            t_set.add(op_dict['tra_id'])
                                            wr_str = 'wr("' + str(key) + '","' + str(op_dict['tra_id']) + '","' + str(
                                                op['var']) + '")\n'
                                            if wr_str not in self.facts:
                                                self.facts.append(wr_str)
                                            has_wr = True
                                            break
                                    if has_wr:
                                        break
                        if op['var'] not in r_nodes:
                            r_nodes[op['var']] = set()
                        r_nodes[op['var']].add(op_dict['tra_id'])
                if op_dict['tra_id'] not in self.txns:
                    self.txns[op_dict['tra_id']] = []
                self.txns[op_dict['tra_id']].extend(current_tra.copy())
                current_tra.clear()
            else:
                current_tra.append(op_dict)

    def get_op(self, op):
        op = op.strip('\n')
        arr = op[2:-1].split(',')
        if arr[1] == '':
            print('Error: empty!')
        return {
            'op_type': op[0],
            'var': arr[0],
            'val': arr[1],
            'client_id': int(arr[2]),
            'tra_id': int(arr[3]),
        }


def store_facts(data):
    facts = FactBase()
    readZero = False
    for i in range(0, len(data)):
        if data[i].startswith('so'):
            temp = data[i].split('"')
            temp1 = temp[1]
            temp2 = temp[3]
            facts.add(So(temp1, temp2))
        if data[i].startswith('wtxn'):
            temp = data[i].split('"')
            temp1 = temp[1]
            temp2 = temp[3]
            facts.add(Wtxn(temp1, temp2))
        if data[i].startswith('wr'):
            temp = data[i].split('"')
            temp1 = temp[1]
            temp2 = temp[3]
            temp3 = temp[5]
            facts.add(Wr(temp1, temp2, temp3))
        if data[i].startswith('wsv'):
            temp = data[i].split('"')
            temp1 = temp[1]
            temp2 = temp[3]
            facts.add(Wsv(temp1, temp2))
    return facts


def detection(url):
    encoding_start = time.time()
    ASP_PROGRAM = "./tools/CausalC+/rules.lp"
    data = linecache.getlines(url)
    facts_plain = ClingoTxnHistory(data)

    facts = store_facts(facts_plain.facts)

    ctrl = Control(unifier=[Bad_CyclicCO])
    ctrl.load(ASP_PROGRAM)
    ctrl.add_facts(facts)
    ctrl.ground([("base", [])])
    solution = None
    return_list = [0, 0]

    encoding_end = time.time()
    encoding_time = int((encoding_end - encoding_start) * 1000)

    print(f"encoding: {encoding_time} ms")

    solving_start = time.time()
    def on_model(model):
        solution = model.facts(atoms=True)
        bad_1 = solution.select(Bad_CyclicCO).get()
        if (len(bad_1)) > 0:
            print('BP!!!')
            return_list[0] = 1
            return_list[1] = 1

    ctrl.solve(on_model=on_model)

    solving_end = time.time()
    solving_time = int((solving_end - solving_start) * 1000)
    print(f"solving: {solving_time} ms")

    return return_list == [1, 1]


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_mono_txn.py <hist_path>")
        sys.exit(1)

    if detection(sys.argv[1]):
        print('REJECT')
    else:
        print('ACCEPT')


if __name__ == "__main__":
    main()
