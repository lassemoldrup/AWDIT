use super::*;

pub struct HistoryChecker<'h, F> {
    pub(super) on_violation: F,
    pub(super) report_mode: ReportMode,
    pub(super) history: &'h History,
    pub(super) any_violation: bool,
}

impl<'h, F: FnMut(&ConsistencyViolation)> HistoryChecker<'h, F> {
    fn check_partial_read_consistency(&mut self) -> Result<(), ConsistencyViolation> {
        for (s_idx, session) in self.history.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                let tid = TransactionId(s_idx, t_idx);
                let mut inter_txn_reads = FxHashSet::default();
                let mut writes = FxHashSet::default();
                let mut write_map = FxHashMap::default();
                for event in &transaction.events {
                    match event {
                        &Event::Read(kv) => {
                            let Some(&w_val) = write_map.get(&kv.key) else {
                                inter_txn_reads.insert(kv);
                                continue;
                            };
                            if w_val == kv.value {
                                continue;
                            }

                            let write_kv = KeyValuePair {
                                key: kv.key,
                                value: w_val,
                            };
                            let violation = if writes.contains(&kv) {
                                ConsistencyViolation::NotMyLastWrite {
                                    tid,
                                    read_event: kv,
                                    last_write: write_kv,
                                }
                            } else {
                                ConsistencyViolation::NotMyOwnWrite {
                                    tid,
                                    read_event: kv,
                                    own_write: write_kv,
                                }
                            };
                            self.report_violation(violation, ReportMode::First)?;
                        }
                        &Event::Write(kv) => {
                            if inter_txn_reads.contains(&kv) {
                                let violation = ConsistencyViolation::FutureRead {
                                    tid,
                                    read_event: kv,
                                };
                                self.report_violation(violation, ReportMode::First)?;
                            }
                            writes.insert(kv);
                            write_map.insert(kv.key, kv.value);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn infer_graph(&mut self) -> Result<WriteReadGraph, ConsistencyViolation> {
        let history = self.history;
        let mut value_map = FxHashMap::default();
        let mut intermediate_writes = FxHashMap::default();
        for (s_idx, session) in history.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                let tid = TransactionId(s_idx, t_idx);
                let mut txn_writes = FxHashMap::default();
                for event in &transaction.events {
                    let &Event::Write(kv) = event else {
                        continue;
                    };
                    let Some(intermediate) = txn_writes.insert(kv.key, kv.value) else {
                        continue;
                    };
                    let intermediate_kv = KeyValuePair {
                        key: kv.key,
                        value: intermediate,
                    };
                    intermediate_writes.insert(intermediate_kv, tid);
                }
                for (key, value) in txn_writes {
                    let kv = KeyValuePair { key, value };
                    if value_map.insert(kv, tid).is_some() {
                        eprintln!("Duplicate writes to {kv}. Using the last one..");
                    }
                }
            }
        }

        let mut graph = WriteReadGraph {
            reads: history
                .sessions
                .iter()
                .map(|s| vec![Vec::new(); s.len()])
                .collect(),
        };

        for (s_idx, session) in history.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                for event in &transaction.events {
                    let &Event::Read(kv) = event else {
                        continue;
                    };

                    let tid = TransactionId(s_idx, t_idx);
                    if self.history.aborted_writes.contains(&kv) {
                        let violation = ConsistencyViolation::AbortedRead { tid, event: kv };
                        self.report_violation(violation, ReportMode::First)?;
                        continue;
                    }

                    if let Some(&writer) = value_map.get(&kv) {
                        if writer != tid {
                            graph.reads[s_idx][t_idx].push((writer, kv));
                        }
                    } else if let Some(&writer) = intermediate_writes.get(&kv) {
                        if writer != tid {
                            // if writer == tid, this is NotMyLastWrite
                            let violation = ConsistencyViolation::IntermediateRead {
                                writer_tid: writer,
                                reader_tid: tid,
                                read_event: kv,
                            };
                            self.report_violation(violation, ReportMode::First)?;
                            graph.reads[s_idx][t_idx].push((writer, kv));
                        }
                    } else {
                        let violation = ConsistencyViolation::ThinAirRead { tid, event: kv };
                        self.report_violation(violation, ReportMode::First)?;
                    }
                }
            }
        }
        Ok(graph)
    }

    fn report_violation(
        &mut self,
        violation: ConsistencyViolation,
        stop_on: ReportMode,
    ) -> Result<(), ConsistencyViolation> {
        (self.on_violation)(&violation);
        self.any_violation = true;
        if self.report_mode <= stop_on {
            Err(violation)
        } else {
            Ok(())
        }
    }

    fn get_repeatable_reads_graph(
        &mut self,
        graph: &WriteReadGraph,
    ) -> Result<RepeatableReadsGraph, ConsistencyViolation> {
        let mut read_map: RepeatableReadsGraph = graph
            .reads
            .iter()
            .map(|s| vec![FxHashMap::default(); s.len()])
            .collect();
        for (s_idx, session) in graph.reads.iter().enumerate() {
            for (t_idx, reads) in session.iter().enumerate() {
                for &(writer_tid, kv) in reads {
                    let writers = read_map[s_idx][t_idx].entry(kv.key).or_default();
                    if writers.len() == 1 {
                        if writers[0].0 == writer_tid {
                            continue;
                        }
                        let (t2, r2_val) = writers[0];
                        let violation = ConsistencyViolation::NonRepeatableRead {
                            reader_tid: TransactionId(s_idx, t_idx),
                            t1: writer_tid,
                            t2,
                            r1: kv,
                            r2: KeyValuePair {
                                key: kv.key,
                                value: r2_val,
                            },
                        };
                        self.report_violation(violation, ReportMode::CausalCycles)?;
                    }
                    writers.push((writer_tid, kv.value));
                }
            }
        }

        Ok(read_map)
    }

    pub fn check_causal(&mut self) -> bool {
        self.any_violation = false;
        if self.check_partial_read_consistency().is_err() {
            return false;
        }
        let Ok(graph) = self.infer_graph() else {
            return false;
        };
        if self.report_mode <= ReportMode::ReadConsistency && self.any_violation {
            return false;
        }

        let history = self.history;

        let mut succ_sessions: FxHashMap<TransactionId, FxHashSet<usize>> = FxHashMap::default();
        for (s_idx, session) in history.sessions.iter().enumerate() {
            for t_idx in 0..session.len() {
                if t_idx < session.len() - 1 {
                    succ_sessions
                        .entry(TransactionId(s_idx, t_idx))
                        .or_default()
                        .insert(s_idx);
                }
                for &(writer_tid, _) in &graph.reads[s_idx][t_idx] {
                    succ_sessions.entry(writer_tid).or_default().insert(s_idx);
                }
            }
        }

        let write_sets = history.get_write_sets();
        let mut commit_order = PartialCommitOrder::new(&history);
        let mut hb: FxHashMap<TransactionId, VectorClock> = FxHashMap::default();
        let mut writes_per_key: FxHashMap<Key, BTreeMap<usize, Vec<usize>>> = FxHashMap::default();

        let dfs_res = graph.dfs(|t3| {
            let mut t3_vc = VectorClock::new_min(history.sessions.len());
            for pred in graph.rev_hb_edges(t3) {
                let Some(pred_vc) = hb.get(&pred) else {
                    continue;
                };
                t3_vc.join(pred_vc);
                t3_vc.join1(pred.0, pred.1 as i32);
                let pred_succ_sessions = succ_sessions.get_mut(&pred).expect("pred is in hb");
                pred_succ_sessions.remove(&t3.0);
                if pred_succ_sessions.is_empty() {
                    hb.remove(&pred);
                    succ_sessions.remove(&pred);
                }
            }

            let t3_reads = &graph.reads[t3.0][t3.1];
            let t3_read_map = to_read_map(t3_reads);
            let mut prev_writers = FxHashMap::default();
            for &(t1, kv) in t3_reads {
                for (&t2_s_idx, writes) in &writes_per_key[&kv.key] {
                    let Ok(last_pred) = usize::try_from(t3_vc[t2_s_idx]) else {
                        // If -1, no predecessors in t2's session
                        continue;
                    };
                    // Find the last write to x in t2's session that is less than or equal to
                    // last_pred
                    let Some(last_write) =
                        writes.partition_point(|&i| i <= last_pred).checked_sub(1)
                    else {
                        continue;
                    };
                    let t2_t_idx = writes[last_write];
                    let t2 = TransactionId(t2_s_idx, t2_t_idx);
                    if t2 == t1 {
                        continue;
                    }

                    // TODO: break out early?
                    // else if hb[t2_s_idx][t2_t_idx][t1.0] >= t1.1 as isize {
                    // }
                    if let Some(&read_y) = prev_writers.get(&t2) {
                        let justification = CoJustificationKind::NonMonotonic(read_y);
                        commit_order.add_edge(t1, t2, t3, kv, justification);
                    } else if t2_s_idx == t3.0 && t2_t_idx < t3.1 {
                        let justification = CoJustificationKind::FracturedSo;
                        commit_order.add_edge(t1, t2, t3, kv, justification);
                    } else if let Some(read_ys) = t3_read_map.get(&t2) {
                        let justification = CoJustificationKind::FracturedWr(read_ys[0]);
                        commit_order.add_edge(t1, t2, t3, kv, justification);
                    } else {
                        let justification = CoJustificationKind::Causal;
                        commit_order.add_edge(t1, t2, t3, kv, justification);
                    }
                }
                prev_writers.insert(t1, kv);
            }

            for &x in &write_sets[t3.0][t3.1] {
                writes_per_key
                    .entry(x)
                    .or_default()
                    .entry(t3.0)
                    .or_default()
                    .push(t3.1);
            }
            hb.insert(t3, t3_vc);
        });
        if let Err(cycle) = dfs_res {
            if let Err(_) =
                self.report_violation(ConsistencyViolation::Cycle(cycle), ReportMode::CausalCycles)
            {
                return false;
            }
        }

        // Check for cycles in the reverse commit order
        commit_order.find_cycles(&graph, &mut self.on_violation) && !self.any_violation
    }

    pub fn check_read_atomic(&mut self) -> bool {
        self.any_violation = false;
        if self.check_partial_read_consistency().is_err() {
            return false;
        }
        let Ok(graph) = self.infer_graph() else {
            return false;
        };
        if self.report_mode <= ReportMode::ReadConsistency && self.any_violation {
            return false;
        }

        if self.report_mode == ReportMode::CausalCycles {
            if let Err(cycle) = graph.dfs(|_| {}) {
                let _ = self
                    .report_violation(ConsistencyViolation::Cycle(cycle), ReportMode::CausalCycles);
                return false;
            }
        }

        let Ok(repeatable_reads_graph) = self.get_repeatable_reads_graph(&graph) else {
            return false;
        };

        let history = self.history;
        let write_sets = history.get_write_sets();

        let mut commit_order = PartialCommitOrder::new(&history);
        for (t3_s_idx, sess_reads) in graph.reads.iter().enumerate() {
            let mut last_writes_per_key = FxHashMap::default();
            for (t3_t_idx, t3_writers) in sess_reads.iter().enumerate() {
                let t3 = TransactionId(t3_s_idx, t3_t_idx);
                let t3_writes = &write_sets[t3_s_idx][t3_t_idx];
                let t3_read_map = to_read_map(t3_writers);
                for &(t1, kv) in t3_writers {
                    let Some(&t2) = last_writes_per_key.get(&kv.key) else {
                        continue;
                    };
                    if t2 == t1 || t3_read_map.contains_key(&t2) {
                        // If we read from t2, we handle it as a read, to know if it is
                        // non-monotonic or fractured
                        continue;
                    }
                    commit_order.add_edge(t1, t2, t3, kv, CoJustificationKind::FracturedSo);
                }

                let po_min_max_map = history.get_po_min_max_map(t3);
                for (t2, t2_reads) in t3_read_map.into_iter() {
                    for (kv, t1) in intersect_map(
                        &repeatable_reads_graph[t3_s_idx][t3_t_idx],
                        &write_sets[t2.0][t2.1],
                    )
                    .flat_map(|(&x, t1s)| {
                        t1s.iter()
                            .map(move |&(t1, value)| (KeyValuePair { key: x, value }, t1))
                    })
                    .filter(|&(_, t1)| t1 != t2)
                    {
                        let read_y = t2_reads[0];
                        if po_min_max_map[&read_y].0 < po_min_max_map[&kv].1 {
                            let justification = CoJustificationKind::NonMonotonic(read_y);
                            commit_order.add_edge(t1, t2, t3, kv, justification);
                        } else {
                            let justifcation = CoJustificationKind::FracturedWr(read_y);
                            commit_order.add_edge(t1, t2, t3, kv, justifcation);
                        }
                    }
                }
                for &k in t3_writes {
                    last_writes_per_key.insert(k, t3);
                }
            }
        }

        // Check for cycles in the reverse commit order
        commit_order.find_cycles(&graph, &mut self.on_violation) && !self.any_violation
    }

    pub fn check_read_committed(&mut self) -> bool {
        self.any_violation = false;
        if self.check_partial_read_consistency().is_err() {
            return false;
        }
        let Ok(graph) = self.infer_graph() else {
            return false;
        };
        if self.report_mode <= ReportMode::ReadConsistency && self.any_violation {
            return false;
        }

        if self.report_mode == ReportMode::CausalCycles {
            if let Err(cycle) = graph.dfs(|_| {}) {
                let _ = self
                    .report_violation(ConsistencyViolation::Cycle(cycle), ReportMode::CausalCycles);
                return false;
            }
        }

        let history = self.history;
        let write_sets = history.get_write_sets();

        let mut commit_order = PartialCommitOrder::new(&history);
        for (t3_s_idx, session) in graph.reads.iter().enumerate() {
            for (t3_t_idx, t3_writers) in session.iter().enumerate() {
                let mut read_txns = FxHashSet::default();
                let mut first_txn_reads = FxHashSet::default();
                for &(t2, kv) in t3_writers {
                    if read_txns.insert(t2) {
                        first_txn_reads.insert(kv);
                    }
                }
                // Stores per key the earliest (last seen) two writers and the values read from
                // them
                let mut earliest_writer_per_loc: FxHashMap<
                    Key,
                    (Option<(TransactionId, Value)>, (TransactionId, Value)),
                > = FxHashMap::default();
                for &(t2, kv) in t3_writers.iter().rev() {
                    if first_txn_reads.contains(&kv) {
                        for (&x, &(w1, w2)) in
                            intersect_map(&earliest_writer_per_loc, &write_sets[t2.0][t2.1])
                        {
                            let (t1, value) = if w2.0 != t2 {
                                w2
                            } else if let Some(w1) = w1 {
                                w1
                            } else {
                                continue;
                            };
                            let t3 = TransactionId(t3_s_idx, t3_t_idx);
                            let kv_x = KeyValuePair { key: x, value };
                            let justification = CoJustificationKind::NonMonotonic(kv);
                            commit_order.add_edge(t1, t2, t3, kv_x, justification);
                        }
                    }
                    earliest_writer_per_loc
                        .entry(kv.key)
                        .and_modify(|e| {
                            *e = (Some(e.1), (t2, kv.value));
                        })
                        .or_insert((None, (t2, kv.value)));
                }
            }
        }

        // Check for cycles in the reverse commit order
        commit_order.find_cycles(&graph, &mut self.on_violation) && !self.any_violation
    }
}

fn to_read_map(
    reads: &[(TransactionId, KeyValuePair)],
) -> FxHashMap<TransactionId, Vec<KeyValuePair>> {
    let mut map: FxHashMap<_, Vec<_>> = FxHashMap::default();
    for &(tid, kv) in reads {
        map.entry(tid).or_default().push(kv);
    }
    map
}

impl Display for History {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for (s_idx, session) in self.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                for event in &transaction.events {
                    writeln!(f, "{event}")?;
                }
                if t_idx < session.len() - 1 {
                    writeln!(f, "-----")?;
                }
            }
            if s_idx < self.sessions.len() - 1 {
                writeln!(f, "=====")?;
            }
        }
        Ok(())
    }
}

struct PartialCommitOrder {
    rev_order: Vec<Vec<Vec<(TransactionId, CoJustification)>>>,
}

impl PartialCommitOrder {
    fn new(shape: &History) -> Self {
        Self {
            rev_order: shape
                .sessions
                .iter()
                .map(|s| vec![Vec::new(); s.len()])
                .collect(),
        }
    }

    fn add_edge(
        &mut self,
        t1: TransactionId,
        t2: TransactionId,
        t3: TransactionId,
        x: KeyValuePair,
        kind: CoJustificationKind,
    ) {
        self.rev_order[t1.0][t1.1].push((t2, CoJustification { t3, kv: x, kind }));
    }

    fn find_cycles(
        &self,
        graph: &WriteReadGraph,
        on_violation: &mut impl FnMut(&ConsistencyViolation),
    ) -> bool {
        let mut success = true;
        let mut state: Vec<_> = graph
            .reads
            .iter()
            .map(|s| vec![TarjanState::default(); s.len()])
            .collect();
        let mut next_index = 0;
        let mut stack = Vec::new();
        for (s_idx, session) in graph.reads.iter().enumerate() {
            let t_idx = session.len() - 1;
            if state[s_idx][t_idx].index != usize::MAX {
                continue;
            }
            success = self.tarjan_visit(
                TransactionId(s_idx, t_idx),
                graph,
                &mut state,
                &mut next_index,
                &mut stack,
                on_violation,
            ) && success;
        }
        success
    }

    fn tarjan_visit(
        &self,
        tid: TransactionId,
        graph: &WriteReadGraph,
        state: &mut Vec<Vec<TarjanState>>,
        next_index: &mut usize,
        stack: &mut Vec<TransactionId>,
        on_violation: &mut impl FnMut(&ConsistencyViolation),
    ) -> bool {
        let mut success = true;
        let node_state = &mut state[tid.0][tid.1];
        node_state.index = *next_index;
        node_state.low_link = *next_index;
        *next_index += 1;
        stack.push(tid);
        node_state.on_stack = true;

        let index = node_state.index;
        let mut low_link = node_state.low_link;

        for (tid2, _) in self.rev_neighbours(tid, graph) {
            let tid2_node_state = state[tid2.0][tid2.1];
            if tid2_node_state.index == usize::MAX {
                success = self.tarjan_visit(tid2, graph, state, next_index, stack, on_violation)
                    && success;
                low_link = low_link.min(state[tid2.0][tid2.1].low_link);
            } else if tid2_node_state.on_stack {
                low_link = low_link.min(tid2_node_state.low_link);
            }
        }

        state[tid.0][tid.1].low_link = low_link;
        if index == state[tid.0][tid.1].low_link {
            let rev_idx = stack
                .iter()
                .rev()
                .position(|&t| t == tid)
                .expect("tid should be in stack");
            if rev_idx == 0 {
                // Singleton SCC
                stack.pop();
                state[tid.0][tid.1].on_stack = false;
                return success;
            }
            let scc_start = stack.len() - rev_idx - 1;
            self.report_scc(&stack[scc_start..], graph, on_violation);
            for &tid2 in &stack[scc_start..] {
                state[tid2.0][tid2.1].on_stack = false;
            }
            stack.truncate(scc_start);
            success = false;
        }
        success
    }

    fn rev_neighbours<'g>(
        &self,
        tid: TransactionId,
        graph: &'g WriteReadGraph,
    ) -> impl Iterator<Item = (TransactionId, Option<CoJustification>)> + Captures<'_> + Captures<'g>
    {
        graph.rev_hb_edges(tid).map(|t| (t, None)).chain(
            self.rev_order[tid.0][tid.1]
                .iter()
                .map(|&(t, j)| (t, Some(j))),
        )
    }

    fn report_scc(
        &self,
        scc: &[TransactionId],
        graph: &WriteReadGraph,
        on_violation: &mut impl FnMut(&ConsistencyViolation),
    ) {
        let root = scc[0];
        let scc_set: FxHashSet<_> = scc.iter().copied().collect();
        let mut visited = FxHashSet::default();
        let mut parent = FxHashMap::default();
        let mut queue = VecDeque::from([root]);
        'queue_loop: while let Some(tid) = queue.pop_front() {
            for (tid2, just) in self.rev_neighbours(tid, graph) {
                if !scc_set.contains(&tid2) || !visited.insert(tid2) {
                    continue;
                }
                parent.insert(tid2, (tid, just));
                if tid2 == root {
                    break 'queue_loop;
                }
                queue.push_back(tid2);
            }
        }

        let cycle = graph.build_cycle(root, |tid| parent[&tid]);
        on_violation(&ConsistencyViolation::Cycle(cycle));
    }
}

#[derive(Clone, Copy)]
struct TarjanState {
    index: usize,
    low_link: usize,
    on_stack: bool,
}

impl Default for TarjanState {
    fn default() -> Self {
        Self {
            index: usize::MAX,
            low_link: usize::MAX,
            on_stack: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CoJustification {
    t3: TransactionId,
    kv: KeyValuePair,
    kind: CoJustificationKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoJustificationKind {
    NonMonotonic(KeyValuePair),
    FracturedSo,
    FracturedWr(KeyValuePair),
    Causal,
}

impl PartialOrd for CoJustificationKind {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }
        match (self, other) {
            (CoJustificationKind::NonMonotonic(_), CoJustificationKind::NonMonotonic(_)) => None,
            (CoJustificationKind::NonMonotonic(_), _) => Some(Ordering::Less),
            (
                CoJustificationKind::FracturedSo | CoJustificationKind::FracturedWr(_),
                CoJustificationKind::NonMonotonic(_),
            ) => Some(Ordering::Greater),
            (
                CoJustificationKind::FracturedSo | CoJustificationKind::FracturedWr(_),
                CoJustificationKind::FracturedSo | CoJustificationKind::FracturedWr(_),
            ) => None,
            (
                CoJustificationKind::FracturedSo | CoJustificationKind::FracturedWr(_),
                CoJustificationKind::Causal,
            ) => Some(Ordering::Less),
            (CoJustificationKind::Causal, _) => Some(Ordering::Greater),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum EdgeType {
    Session,
    WriteRead,
    CommitOrder(CoJustification),
}

impl EdgeType {
    fn from_justification(
        t1: TransactionId,
        t2: TransactionId,
        just: Option<CoJustification>,
    ) -> Self {
        if t1.0 == t2.0 && t2.1 > t1.1 {
            Self::Session
        } else if let Some(just) = just {
            Self::CommitOrder(just)
        } else {
            Self::WriteRead
        }
    }
}

impl Display for EdgeType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            EdgeType::Session => write!(f, "SO"),
            EdgeType::WriteRead => write!(f, "WR"),
            EdgeType::CommitOrder(_) => write!(f, "CO"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchState {
    NotSeen,
    Marked,
    Seen,
}

#[derive(Debug)]
enum DfsStackEntry {
    Pre(TransactionId),
    Post(TransactionId),
}

type RepeatableReadsGraph = Vec<Vec<FxHashMap<Key, SmallVec<[(TransactionId, Value); 1]>>>>;

#[derive(Debug)]
struct WriteReadGraph {
    reads: Vec<Vec<Vec<(TransactionId, KeyValuePair)>>>,
}

impl WriteReadGraph {
    fn dfs(&self, mut post_action: impl FnMut(TransactionId)) -> Result<(), ViolatingCycle> {
        let mut search_state: Vec<_> = self
            .reads
            .iter()
            .map(|s| vec![SearchState::NotSeen; s.len()])
            .collect();
        let mut parent: Vec<_> = self.reads.iter().map(|s| vec![None; s.len()]).collect();
        // Iterate edges backwards to get a forward top order
        for (s_idx, session) in self.reads.iter().enumerate() {
            let t_idx = session.len() - 1;
            if search_state[s_idx][t_idx] == SearchState::Seen {
                continue;
            }

            let mut stack = vec![DfsStackEntry::Pre(TransactionId(s_idx, t_idx))];
            while let Some(entry) = stack.pop() {
                match entry {
                    DfsStackEntry::Pre(tid) => {
                        if search_state[tid.0][tid.1] == SearchState::Seen {
                            continue;
                        }
                        search_state[tid.0][tid.1] = SearchState::Marked;
                        stack.push(DfsStackEntry::Post(tid));

                        for tid2 in self.rev_hb_edges(tid) {
                            match search_state[tid2.0][tid2.1] {
                                SearchState::NotSeen => {
                                    stack.push(DfsStackEntry::Pre(tid2));
                                    parent[tid2.0][tid2.1] = Some(tid);
                                }
                                SearchState::Marked => {
                                    // Cycle detected
                                    parent[tid2.0][tid2.1] = Some(tid);
                                    return Err(self
                                        .build_cycle(tid2, |t| (parent[t.0][t.1].unwrap(), None)));
                                }
                                SearchState::Seen => {
                                    continue;
                                }
                            }
                        }
                    }
                    DfsStackEntry::Post(tid) => {
                        search_state[tid.0][tid.1] = SearchState::Seen;
                        post_action(tid);
                    }
                }
            }
        }

        Ok(())
    }

    fn build_cycle(
        &self,
        root: TransactionId,
        parent: impl Fn(TransactionId) -> (TransactionId, Option<CoJustification>),
    ) -> ViolatingCycle {
        let mut cycle = vec![root];
        let mut edges = vec![];
        // If we ever go from (s, t) to (s, t + d), we can remove all in-between steps
        let mut index_in_session: FxHashMap<_, _> = [(root.0, (root.1, 1))].into_iter().collect();
        let mut prev = root;
        let (mut next, mut just) = parent(root);
        loop {
            if next == root {
                cycle.push(root);
                edges.push(EdgeType::from_justification(prev, next, just));
                break;
            }

            match index_in_session.entry(next.0) {
                Entry::Occupied(e) => {
                    let (t_idx, len) = e.get();
                    assert_ne!(next.1, *t_idx);
                    if next.1 > *t_idx {
                        // Compact
                        cycle.truncate(*len);
                        edges.truncate(*len - 1);
                    } else {
                        // Shorter cycle found
                        cycle = cycle.split_off(*len - 1);
                        cycle.push(next);
                        cycle.push(TransactionId(next.0, *t_idx));
                        edges = edges.split_off(*len - 1);
                        edges.push(EdgeType::from_justification(prev, next, just));
                        edges.push(EdgeType::Session);
                        break;
                    }
                }
                Entry::Vacant(e) => {
                    e.insert((next.1, cycle.len() + 1));
                }
            }
            cycle.push(next);
            edges.push(EdgeType::from_justification(prev, next, just));
            prev = next;
            (next, just) = parent(next);
        }
        ViolatingCycle { cycle, edges }
    }

    fn rev_hb_edges(
        &self,
        TransactionId(s_idx, t_idx): TransactionId,
    ) -> impl Iterator<Item = TransactionId> + Captures<'_> {
        let session_pred = t_idx
            .checked_sub(1)
            .map(|t_idx| TransactionId(s_idx, t_idx))
            .into_iter();
        let reads = self.reads[s_idx][t_idx]
            .iter()
            .map(|(writer, _)| *writer)
            // Get rid of parallel edges
            .filter(move |t| t.0 != s_idx || t.1 + 1 != t_idx)
            .dedup();
        session_pred.chain(reads)
    }
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum ConsistencyViolation {
    #[error("Transaction {tid} reads {event} out of thin air")]
    ThinAirRead {
        tid: TransactionId,
        event: KeyValuePair,
    },
    #[error("Transaction {tid} reads aborted {event}")]
    AbortedRead {
        tid: TransactionId,
        event: KeyValuePair,
    },
    #[error("Transaction {tid} reads {read_event} from later in the same transaction")]
    FutureRead {
        tid: TransactionId,
        read_event: KeyValuePair,
    },
    #[error("Transaction {tid} reads {read_event} instead of own write {own_write}")]
    NotMyOwnWrite {
        tid: TransactionId,
        read_event: KeyValuePair,
        own_write: KeyValuePair,
    },
    #[error("Transaction {tid} reads {read_event} instead of last write {last_write}")]
    NotMyLastWrite {
        tid: TransactionId,
        read_event: KeyValuePair,
        last_write: KeyValuePair,
    },
    #[error(
        "Transaction {reader_tid} reads intermediate {read_event} from {writer_tid} instead of last write"
    )]
    IntermediateRead {
        writer_tid: TransactionId,
        reader_tid: TransactionId,
        read_event: KeyValuePair,
    },
    #[error(
        "Non-repeatable read: Transaction {reader_tid} reads {r1} from {t1} and {r2} from {t2}"
    )]
    NonRepeatableRead {
        reader_tid: TransactionId,
        t1: TransactionId,
        t2: TransactionId,
        r1: KeyValuePair,
        r2: KeyValuePair,
    },
    #[error("{0}")]
    Cycle(ViolatingCycle),
}

#[derive(Clone, Debug)]
pub struct ViolatingCycle {
    cycle: Vec<TransactionId>,
    edges: Vec<EdgeType>,
}

impl Display for ViolatingCycle {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut co_kind = None;
        let mut cycle_str = String::new();
        let mut justification_str = String::new();
        for (i, (&tid, &edge)) in self.cycle.iter().zip(&self.edges).enumerate() {
            write!(&mut cycle_str, "{tid} -> ").unwrap();
            let next = self.cycle[i + 1];
            if let EdgeType::CommitOrder(just) = edge {
                let worst = co_kind.get_or_insert(just.kind);
                if just.kind < *worst {
                    *worst = just.kind;
                }
                write!(&mut justification_str, "{tid} -> {next}: Inferred CO: ").unwrap();
                write!(
                    &mut justification_str,
                    "{} reads {} from {next}",
                    just.t3, just.kv,
                )
                .unwrap();
                match just.kind {
                    CoJustificationKind::NonMonotonic(kv) => {
                        writeln!(
                            &mut justification_str,
                            " after it reads {kv} from {tid}, which also writes {}",
                            just.kv.key
                        )
                        .unwrap();
                    }
                    CoJustificationKind::FracturedSo => writeln!(
                        &mut justification_str,
                        " later in the same session as {tid}, which also writes {}",
                        just.kv.key
                    )
                    .unwrap(),
                    CoJustificationKind::FracturedWr(kv) => writeln!(
                        &mut justification_str,
                        " as well as {kv} from {tid}, which also writes {}",
                        just.kv.key
                    )
                    .unwrap(),
                    CoJustificationKind::Causal => writeln!(
                        &mut justification_str,
                        " and happens after {tid}, which also writes {}",
                        just.kv.key
                    )
                    .unwrap(),
                }
            } else {
                writeln!(&mut justification_str, "{tid} -> {next}: {edge}").unwrap();
            }
        }
        writeln!(cycle_str, "{}", self.cycle[0]).unwrap();

        if let Some(co_kind) = co_kind {
            match co_kind {
                CoJustificationKind::NonMonotonic(_) => write!(f, "Non-monotonic read(s): ")?,
                CoJustificationKind::FracturedSo | CoJustificationKind::FracturedWr(_) => {
                    write!(f, "Fractured read(s): ")?;
                }
                CoJustificationKind::Causal => write!(f, "Causally inconsistent read(s): ")?,
            }
        } else {
            write!(f, "HB cycle: ")?;
        }
        writeln!(f, "{cycle_str}")?;
        write!(f, "{justification_str}")
    }
}

#[cfg(test)]
mod tests {
    use test_generator::test_resources;

    use crate::ReportMode;

    use super::History;

    #[test_resources("histories/tests/causal/**/*.txt")]
    fn test_causal(file: &str) {
        let history = History::parse_test_history(file).unwrap();
        let mut checker = history.checker(ReportMode::First, |_| {});
        assert!(checker.check_causal());
        assert!(checker.check_read_atomic());
        assert!(checker.check_read_committed());
    }

    #[test_resources("histories/tests/read-atomic/**/*.txt")]
    fn test_read_atomic(file: &str) {
        let history = History::parse_test_history(file).unwrap();
        let mut checker = history.checker(ReportMode::First, |_| {});
        assert!(!checker.check_causal());
        assert!(checker.check_read_atomic());
        assert!(checker.check_read_committed());
    }

    #[test_resources("histories/tests/read-committed/**/*.txt")]
    fn test_read_committed(file: &str) {
        let history = History::parse_test_history(file).unwrap();
        let mut checker = history.checker(ReportMode::First, |violation| println!("{violation}"));
        assert!(!checker.check_causal());
        assert!(!checker.check_read_atomic());
        assert!(checker.check_read_committed());
    }

    #[test_resources("histories/tests/none/**/*.txt")]
    fn test_inconsistent(file: &str) {
        let history = History::parse_test_history(file).unwrap();
        let mut checker = history.checker(ReportMode::First, |_| {});
        assert!(!checker.check_causal());
        assert!(!checker.check_read_atomic());
        assert!(!checker.check_read_committed());
    }
}
