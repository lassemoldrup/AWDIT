import copy

from monosat import *

sys.setrecursionlimit(1000000)


class DiGraph:
    def __init__(self):
        self.adj_map = {}

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

    def has_cycle(self):
        for key in list(self.adj_map.keys()):
            reachable = set()
            if self.dfs_util_reach(key, key, reachable):
                return True
        return False

    def dfs_util_reach(self, s, u, reachable):
        if u in self.adj_map:
            for node in self.adj_map[u]:
                if node == s:
                    return True
                elif node in reachable:
                    continue
                else:
                    reachable.add(node)
                    if self.dfs_util_reach(s, node, reachable):
                        return True
        return False

    def dfs_util_all(self, u, reachable):
        if u in self.adj_map:
            for node in self.adj_map[u]:
                if node in reachable:
                    continue
                reachable.add(node)
                self.dfs_util_all(node, reachable)

    def take_closure(self):
        clone_map = self.adj_map.copy()
        for node in self.adj_map:
            reachable = set()
            self.dfs_util_all(node, reachable)
            clone_map[node] = reachable
        self.adj_map = clone_map

    # def take_closure(self):
    #     def dfs(u):
    #         if u in visited:
    #             return
    #         visited.add(u)
    #         if u not in self.adj_map:
    #             return
    #         for node in self.adj_map[u]:
    #             clone_map[u].add(node)
    #             if node in self.adj_map:
    #                 dfs(node)
    #                 clone_map[u].update(clone_map[node])
    #     visited = set()
    #     clone_map = copy.deepcopy(self.adj_map)
    #     for node in self.adj_map:
    #         dfs(node)
    #     self.adj_map = clone_map

    def union_with(self, g):
        for key, value in g.adj_map.items():
            if key in self.adj_map:
                self.adj_map[key] = self.adj_map[key].union(value)
            else:
                self.adj_map[key] = value


class MonosatTxnGraph:
    def __init__(self, ops):
        self.so = DiGraph()
        self.vis = DiGraph()
        self.g = Graph()
        self.wr_rel = {}
        # self.nodes_in_graph = []
        self.nodes_in_graph = {}  # txn_id: xx
        self.txns = {}
        client_in_so = {}
        r_nodes = {}
        current_tra = []
        for i in range(len(ops)):
            op_dict = self.get_op(ops[i])
            if i == len(ops) - 1 or self.get_op(ops[i + 1])['tra_id'] != op_dict['tra_id']:
                self.nodes_in_graph[op_dict['tra_id']] = self.g.addNode()
                if op_dict['client_id'] in client_in_so:
                    self.so.add_edge(client_in_so[op_dict['client_id']], op_dict['tra_id'])
                    Assert(self.g.addEdge(self.nodes_in_graph[client_in_so[op_dict['client_id']]],
                                          self.nodes_in_graph[op_dict['tra_id']]))
                client_in_so[op_dict['client_id']] = op_dict['tra_id']
                current_tra.append(op_dict)
                for op in current_tra:
                    if op['op_type'] == 'w':
                        if op['var'] in self.wr_rel:
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
                                            self.wr_rel[op['var']].add_edge(op_dict['tra_id'], key)
                                            Assert(
                                                self.g.addEdge(self.nodes_in_graph[op_dict['tra_id']],
                                                               self.nodes_in_graph[key]))
                                            break
                    else:
                        has_wr = False
                        if op['var'] in self.wr_rel:
                            for key, t_set in self.wr_rel[op['var']].adj_map.items():
                                if key != op_dict['tra_id']:
                                    for node in self.txns[key]:
                                        if node['val'] == op['val'] and node['var'] == op['var'] and node[
                                            'op_type'] == 'w':
                                            t_set.add(op_dict['tra_id'])
                                            Assert(
                                                self.g.addEdge(self.nodes_in_graph[key],
                                                               self.nodes_in_graph[op_dict['tra_id']]))
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
        self.vis = copy.deepcopy(self.so)
        self.so.take_closure()

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

    def get_wr(self):
        wr = DiGraph()
        for key, digraph in self.wr_rel.items():
            wr.union_with(digraph)
        return wr

    def vis_includes(self, g):
        self.vis.union_with(g)

    def vis_is_trans(self):
        self.vis.take_closure()

    def casual_ww(self):
        for x, wr_x in self.wr_rel.items():
            for t1, t3s in wr_x.adj_map.items():
                for t2 in list(wr_x.adj_map):
                    if t1 != t2:
                        has_edge = False
                        if self.vis.has_edge(t2, t1):
                            has_edge = True
                        else:
                            for t3 in t3s:
                                if self.vis.has_edge(t2, t3):
                                    has_edge = True
                                    break
                        if has_edge:
                            Assert(self.g.addEdge(self.nodes_in_graph[t2], self.nodes_in_graph[t1]))

    def check_cycle(self):
        Assert(self.g.acyclic(True))
        result = Solve()
        Monosat().newSolver()
        return not result
