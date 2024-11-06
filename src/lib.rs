use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::VecDeque;
use std::fmt::{self, Display, Formatter, Write};
use std::mem;

#[cfg(feature = "dbcop")]
use dbcop::db::history::HistParams;

use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use util::{intersect_map, Captures, GetTwoMut};
use vector_clock::VectorClock;

pub mod fenwick;
pub mod formats;
pub mod partial_order;
pub mod util;
pub mod vector_clock;

pub struct History {
    pub sessions: Vec<Vec<Transaction>>,
    pub aborted_writes: FxHashSet<KeyValuePair>,
}

impl History {
    pub fn checker<R: ConsistencyReport>(&self) -> HistoryChecker<R> {
        HistoryChecker {
            report: R::default(),
            history: self,
        }
    }

    pub fn stats(&self) -> HistoryStats {
        HistoryStats {
            num_sessions: self.sessions.len(),
            num_transactions: self.sessions.iter().map(|s| s.len()).sum(),
            num_events: self
                .sessions
                .iter()
                .flat_map(|s| s.iter().map(|t| t.events.len()))
                .sum(),
            num_keys: self
                .sessions
                .iter()
                .flat_map(|s| s.iter().flat_map(|t| t.events.iter()))
                .map(|e| e.key())
                .unique()
                .count(),
        }
    }

    fn get_write_sets(&self) -> Vec<Vec<FxHashSet<Key>>> {
        let mut write_sets: Vec<Vec<FxHashSet<_>>> = self
            .sessions
            .iter()
            .map(|sess| vec![FxHashSet::default(); sess.len()])
            .collect();
        for (s_idx, session) in self.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                let writes = &mut write_sets[s_idx][t_idx];
                for &e in &transaction.events {
                    match e {
                        Event::Read(_) => {}
                        Event::Write(kv) => {
                            writes.insert(kv.key);
                        }
                    }
                }
            }
        }
        write_sets
    }

    fn get_writes_per_key(&self) -> FxHashMap<Key, Vec<Vec<usize>>> {
        let mut writes = FxHashMap::default();
        for (s_idx, session) in self.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                for event in transaction.events.iter().rev() {
                    let &Event::Write(kv) = event else {
                        continue;
                    };
                    let session_writes = &mut writes
                        .entry(kv.key)
                        .or_insert_with(|| vec![Vec::new(); self.sessions.len()])[s_idx];
                    if let Some(&t2_idx) = session_writes.last() {
                        if t2_idx == t_idx {
                            // Multiple writes in the same transaction to the same location
                            continue;
                        }
                    }
                    session_writes.push(t_idx);
                }
            }
        }
        writes
    }

    fn get_po_min_max_map(&self, tid: TransactionId) -> FxHashMap<KeyValuePair, (usize, usize)> {
        let mut map = FxHashMap::default();
        for (po_idx, &event) in self.sessions[tid.0][tid.1].events.iter().enumerate() {
            if let Event::Read(kv) = event {
                let (_, max) = map.entry(kv).or_insert((po_idx, po_idx));
                *max = po_idx;
            }
        }
        map
    }

    pub fn strip_64th_bit(&mut self) {
        for session in &mut self.sessions {
            for transaction in session {
                for event in &mut transaction.events {
                    let kv = match event {
                        Event::Read(kv) => kv,
                        Event::Write(kv) => kv,
                    };
                    kv.key.0 &= 0x7FFF_FFFF_FFFF_FFFF;
                    kv.value.0 &= 0x7FFF_FFFF_FFFF_FFFF;
                }
            }
        }
        let aborted_writes = self
            .aborted_writes
            .iter()
            .map(|kv| KeyValuePair {
                key: Key(kv.key.0 & 0x7FFF_FFFF_FFFF_FFFF),
                value: Value(kv.value.0 & 0x7FFF_FFFF_FFFF_FFFF),
            })
            .collect();
        self.aborted_writes = aborted_writes;
    }

    pub fn fix_thin_air_reads(&mut self) {
        let writes: FxHashSet<_> = self
            .sessions
            .iter()
            .flat_map(|s| s.iter().flat_map(|t| &t.events))
            .filter_map(|e| e.is_write().then(|| e.kv()))
            .collect();
        for session in &mut self.sessions {
            for txn in session {
                for event in &mut txn.events {
                    if let Event::Read(kv) = event {
                        if !writes.contains(kv) && !self.aborted_writes.contains(kv) {
                            kv.value = Value(0);
                        }
                    }
                }
            }
        }
    }
}

pub struct HistoryStats {
    pub num_sessions: usize,
    pub num_transactions: usize,
    pub num_events: usize,
    pub num_keys: usize,
}

impl HistoryStats {
    #[cfg(feature = "dbcop")]
    pub fn to_hist_params(&self) -> HistParams {
        HistParams {
            id: 0,
            n_node: 1,
            n_variable: self.num_keys,
            n_transaction: self.num_transactions,
            n_event: self.num_events,
        }
    }
}

impl Display for HistoryStats {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "#Sessions: {}", self.num_sessions)?;
        writeln!(f, "#Txns:     {}", self.num_transactions)?;
        writeln!(f, "#Events:   {}", self.num_events)?;
        writeln!(f, "#Keys:     {}", self.num_keys)
    }
}

pub struct HistoryChecker<'h, R> {
    report: R,
    history: &'h History,
}

impl<'h, R: ConsistencyReport> HistoryChecker<'h, R> {
    fn check_intra_transactional(&mut self) -> Result<(), ConsistencyViolation> {
        macro_rules! report_violation {
            ($violation:expr) => {
                self.report.add_violation($violation.clone());
                if !R::IS_EXHAUSTIVE {
                    return Err($violation);
                }
            };
        }

        for (s_idx, session) in self.history.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                let tid = TransactionId(s_idx, t_idx);
                let mut inter_txn_reads = FxHashSet::default();
                let mut writes = FxHashSet::default();
                let mut write_map = FxHashMap::default();
                for event in &transaction.events {
                    match event {
                        &Event::Read(kv) => {
                            if let Some(&w_val) = write_map.get(&kv.key) {
                                if w_val != kv.value {
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
                                    report_violation!(violation);
                                }
                            } else {
                                inter_txn_reads.insert(kv);
                            }
                        }
                        &Event::Write(kv) => {
                            if inter_txn_reads.contains(&kv) {
                                let violation = ConsistencyViolation::FutureRead {
                                    tid,
                                    read_event: kv,
                                };
                                report_violation!(violation);
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
        macro_rules! report_violation {
            ($violation:expr) => {
                self.report.add_violation($violation.clone());
                if !R::IS_EXHAUSTIVE {
                    return Err($violation);
                }
            };
        }

        let history = self.history;
        let mut value_map = FxHashMap::default();
        let mut intermediate_writes = FxHashMap::default();
        for (s_idx, session) in history.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                let tid = TransactionId(s_idx, t_idx);
                let mut txn_writes = FxHashMap::default();
                for event in &transaction.events {
                    if let &Event::Write(kv) = event {
                        if let Some(intermediate) = txn_writes.insert(kv.key, kv.value) {
                            intermediate_writes.insert(
                                KeyValuePair {
                                    key: kv.key,
                                    value: intermediate,
                                },
                                tid,
                            );
                        }
                    }
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
                    if let &Event::Read(kv) = event {
                        let tid = TransactionId(s_idx, t_idx);
                        if self.history.aborted_writes.contains(&kv) {
                            let violation = ConsistencyViolation::AbortedRead { tid, event: kv };
                            report_violation!(violation);
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
                                report_violation!(violation);
                                graph.reads[s_idx][t_idx].push((writer, kv));
                            }
                        } else {
                            let violation = ConsistencyViolation::ThinAirRead { tid, event: kv };
                            report_violation!(violation);
                        }
                    }
                }
            }
        }
        Ok(graph)
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
                        } else {
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
                            self.report.add_violation(violation.clone());
                            if !R::IS_EXHAUSTIVE {
                                return Err(violation);
                            }
                        }
                    }
                    writers.push((writer_tid, kv.value));
                }
            }
        }

        Ok(read_map)
    }

    pub fn check_causal(&mut self) -> R {
        if self.check_intra_transactional().is_err() {
            return mem::take(&mut self.report);
        }
        let Ok(graph) = self.infer_graph() else {
            return mem::take(&mut self.report);
        };

        let hb = match graph.compute_hb() {
            Ok(hb) => hb,
            Err(cycle) => {
                self.report
                    .add_violation(ConsistencyViolation::Cycle(cycle));
                return mem::take(&mut self.report);
            }
        };

        let history = self.history;
        let writes_per_key = history.get_writes_per_key();

        let mut commit_order = PartialCommitOrder::new(&history);
        for (t3_s_idx, sess_reads) in graph.reads.iter().enumerate() {
            let mut last_writes_per_key = FxHashMap::default();
            for (t3_t_idx, t3_reads) in sess_reads.iter().enumerate() {
                let t3_read_map = to_read_map(t3_reads);
                let mut prev_writers = FxHashMap::default();
                for &(t1, kv) in t3_reads {
                    let last_writes: &mut Vec<isize> = last_writes_per_key
                        .entry(kv.key)
                        .or_insert_with(|| vec![-1; history.sessions.len()]);
                    for (t2_s_idx, last_write) in last_writes.iter_mut().enumerate() {
                        let Ok(last_pred) = usize::try_from(hb[t3_s_idx][t3_t_idx][t2_s_idx])
                        else {
                            // If -1, no predecessors in t2's session
                            continue;
                        };
                        // Find the last write to x in t2's session that is less than or equal to last_pred
                        let writes = &writes_per_key[&kv.key][t2_s_idx];
                        // TOOD: Test binary search
                        for write_idx in 0.max(*last_write)..writes.len() as isize {
                            match writes[write_idx as usize].cmp(&last_pred) {
                                Ordering::Less => *last_write = write_idx,
                                Ordering::Equal => {
                                    *last_write = write_idx;
                                    break;
                                }
                                Ordering::Greater => {
                                    break;
                                }
                            }
                        }
                        if *last_write >= 0 {
                            let t2_t_idx = writes[*last_write as usize];
                            let t2 = TransactionId(t2_s_idx, t2_t_idx);
                            if t2 == t1 {
                                continue;
                            }
                            // TODO: break out early?
                            // else if hb[t2_s_idx][t2_t_idx][t1.0] >= t1.1 as isize {
                            // }
                            let t3 = TransactionId(t3_s_idx, t3_t_idx);
                            if let Some(&read_y) = prev_writers.get(&t2) {
                                commit_order.add_edge(
                                    t1,
                                    t2,
                                    t3,
                                    kv,
                                    CoJustificationKind::NonMonotonic(read_y),
                                );
                            } else if t2_s_idx == t3_s_idx && t2_t_idx < t3_t_idx {
                                commit_order.add_edge(
                                    t1,
                                    t2,
                                    t3,
                                    kv,
                                    CoJustificationKind::FracturedSo,
                                );
                            } else if let Some(read_ys) = t3_read_map.get(&t2) {
                                commit_order.add_edge(
                                    t1,
                                    t2,
                                    t3,
                                    kv,
                                    CoJustificationKind::FracturedWr(read_ys[0]),
                                );
                            } else {
                                commit_order.add_edge(t1, t2, t3, kv, CoJustificationKind::Causal);
                            }
                        }
                    }
                    prev_writers.insert(t1, kv);
                }
            }
        }

        // Check for cycles in the reverse commit order
        commit_order.find_cycles(&graph, &mut self.report);
        mem::take(&mut self.report)
    }

    pub fn check_read_atomic(&mut self) -> R {
        if self.check_intra_transactional().is_err() {
            return mem::take(&mut self.report);
        }
        let Ok(graph) = self.infer_graph() else {
            return mem::take(&mut self.report);
        };

        let Ok(repeatable_reads_graph) = self.get_repeatable_reads_graph(&graph) else {
            return mem::take(&mut self.report);
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
                    if let Some(&t2) = last_writes_per_key.get(&kv.key) {
                        if t2 == t1 || t3_read_map.contains_key(&t2) {
                            // If we read from t2, we handle it as a read, to know if it is non-monotonic or fractured
                            continue;
                        }
                        commit_order.add_edge(t1, t2, t3, kv, CoJustificationKind::FracturedSo);
                    }
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
                            commit_order.add_edge(
                                t1,
                                t2,
                                t3,
                                kv,
                                CoJustificationKind::NonMonotonic(read_y),
                            );
                        } else {
                            commit_order.add_edge(
                                t1,
                                t2,
                                t3,
                                kv,
                                CoJustificationKind::FracturedWr(read_y),
                            );
                        }
                    }
                }
                for &k in t3_writes {
                    last_writes_per_key.insert(k, t3);
                }
            }
        }

        // Check for cycles in the reverse commit order
        commit_order.find_cycles(&graph, &mut self.report);
        mem::take(&mut self.report)
    }

    pub fn check_read_committed(&mut self) -> R {
        if self.check_intra_transactional().is_err() {
            return mem::take(&mut self.report);
        }
        let Ok(graph) = self.infer_graph() else {
            return mem::take(&mut self.report);
        };

        let history = self.history;
        let write_sets = history.get_write_sets();

        let mut commit_order = PartialCommitOrder::new(&history);
        for (t3_s_idx, session) in graph.reads.iter().enumerate() {
            for (t3_t_idx, t3_writers) in session.iter().enumerate() {
                let mut earliest_writer_per_loc: FxHashMap<Key, (TransactionId, Value)> =
                    FxHashMap::default();
                for &(t2, kv) in t3_writers.iter().rev() {
                    for (&x, &(t1, value)) in
                        intersect_map(&earliest_writer_per_loc, &write_sets[t2.0][t2.1])
                    {
                        if t1 != t2 {
                            let t3 = TransactionId(t3_s_idx, t3_t_idx);
                            let kv_x = KeyValuePair { key: x, value };
                            commit_order.add_edge(
                                t1,
                                t2,
                                t3,
                                kv_x,
                                CoJustificationKind::NonMonotonic(kv),
                            );
                        }
                    }
                    earliest_writer_per_loc.insert(kv.key, (t2, kv.value));
                }
            }
        }

        // Check for cycles in the reverse commit order
        commit_order.find_cycles(&graph, &mut self.report);
        mem::take(&mut self.report)
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

    fn find_cycles<R: ConsistencyReport>(&self, graph: &WriteReadGraph, report: &mut R) {
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
            self.tarjan_visit(
                TransactionId(s_idx, t_idx),
                graph,
                &mut state,
                &mut next_index,
                &mut stack,
                report,
            );
        }
    }

    fn tarjan_visit<R: ConsistencyReport>(
        &self,
        tid: TransactionId,
        graph: &WriteReadGraph,
        state: &mut Vec<Vec<TarjanState>>,
        next_index: &mut usize,
        stack: &mut Vec<TransactionId>,
        report: &mut R,
    ) {
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
                self.tarjan_visit(tid2, graph, state, next_index, stack, report);
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
                return;
            }
            let scc_start = stack.len() - rev_idx - 1;
            self.report_scc(&stack[scc_start..], graph, report);
            for &tid2 in &stack[scc_start..] {
                state[tid2.0][tid2.1].on_stack = false;
            }
            stack.truncate(scc_start);
        }
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

    fn report_scc<R: ConsistencyReport>(
        &self,
        scc: &[TransactionId],
        graph: &WriteReadGraph,
        report: &mut R,
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
        report.add_violation(ConsistencyViolation::Cycle(cycle));
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

enum DfsStackEntry {
    Pre(TransactionId),
    Post(TransactionId),
}

#[derive(Clone, Debug)]
pub struct Transaction {
    pub events: Vec<Event>,
}

impl Transaction {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn last(&self) -> Option<&Event> {
        self.events.last()
    }

    pub fn push(&mut self, event: Event) {
        self.events.push(event);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TransactionId(pub usize, pub usize);

impl Display for TransactionId {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Event {
    Read(KeyValuePair),
    Write(KeyValuePair),
}

impl Event {
    pub fn key(self) -> Key {
        match self {
            Event::Read(kv) | Event::Write(kv) => kv.key,
        }
    }

    pub fn value(self) -> Value {
        match self {
            Event::Read(kv) | Event::Write(kv) => kv.value,
        }
    }

    pub fn kv(self) -> KeyValuePair {
        match self {
            Event::Read(kv) | Event::Write(kv) => kv,
        }
    }

    pub fn is_read(self) -> bool {
        matches!(self, Self::Read(_))
    }

    pub fn is_write(self) -> bool {
        matches!(self, Self::Write(_))
    }
}

impl Display for Event {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Event::Read(kv) => write!(f, "r {} {}", kv.key, kv.value),
            Event::Write(kv) => write!(f, "w {} {}", kv.key, kv.value),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct KeyValuePair {
    pub key: Key,
    pub value: Value,
}

impl Display for KeyValuePair {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[k: {}, v: {}]", self.key, self.value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Key(pub usize);

impl Display for Key {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Value(pub usize);

impl Display for Value {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

type RepeatableReadsGraph = Vec<Vec<FxHashMap<Key, SmallVec<[(TransactionId, Value); 1]>>>>;

#[derive(Debug)]
struct WriteReadGraph {
    reads: Vec<Vec<Vec<(TransactionId, KeyValuePair)>>>,
}

impl WriteReadGraph {
    fn compute_hb(&self) -> Result<Vec<Vec<VectorClock>>, ViolatingCycle> {
        let mut hb: Vec<Vec<VectorClock>> = self
            .reads
            .iter()
            .map(|s| vec![VectorClock::new_min(self.reads.len()); s.len()])
            .collect();

        self.dfs(|TransactionId(s_idx, t_idx)| {
            if t_idx != 0 {
                let (pred, succ) = hb[s_idx].get_two_mut(t_idx - 1, t_idx);
                succ.join(pred);
                succ.join1(s_idx, t_idx as isize - 1);
            }
            for (writer, _) in &self.reads[s_idx][t_idx] {
                if writer.0 == s_idx {
                    // Already covered by session predecessor
                    continue;
                }
                let (pred_sess, succ_sess) = hb.get_two_mut(writer.0, s_idx);
                succ_sess[t_idx].join(&pred_sess[writer.1]);
                succ_sess[t_idx].join1(writer.0, writer.1 as isize);
            }
        })?;

        Ok(hb)
    }

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
                    DfsStackEntry::Pre(tid @ TransactionId(s_idx, t_idx)) => {
                        search_state[s_idx][t_idx] = SearchState::Marked;
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

pub trait ConsistencyReport: Default + Display {
    const IS_EXHAUSTIVE: bool;

    fn add_violation(&mut self, violation: ConsistencyViolation);
    fn is_success(&self) -> bool;
}

pub struct WeakestViolationReport(Result<(), ConsistencyViolation>);

impl Default for WeakestViolationReport {
    fn default() -> Self {
        Self(Ok(()))
    }
}

impl Display for WeakestViolationReport {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match &self.0 {
            Ok(_) => writeln!(f, "Consistent."),
            Err(violation) => {
                writeln!(f, "Inconsistent:")?;
                writeln!(f, "{violation}")
            }
        }
    }
}

impl ConsistencyReport for WeakestViolationReport {
    const IS_EXHAUSTIVE: bool = false;

    fn add_violation(&mut self, violation: ConsistencyViolation) {
        self.0 = Err(violation);
    }

    fn is_success(&self) -> bool {
        self.0.is_ok()
    }
}

#[derive(Default)]
pub struct FullViolationReport(Vec<ConsistencyViolation>);

impl Display for FullViolationReport {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.is_success() {
            writeln!(f, "Consistent.")?
        } else {
            writeln!(f, "Inconsistent:")?;
            for violation in &self.0 {
                writeln!(f, "{violation}")?;
            }
        }
        Ok(())
    }
}

impl ConsistencyReport for FullViolationReport {
    const IS_EXHAUSTIVE: bool = true;

    fn add_violation(&mut self, violation: ConsistencyViolation) {
        self.0.push(violation);
    }

    fn is_success(&self) -> bool {
        self.0.is_empty()
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
    #[error("Transaction {reader_tid} reads intermediate {read_event} from {writer_tid} instead of last write")]
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
    use std::fs;

    use rustc_hash::FxHashSet;
    use test_generator::test_resources;

    use crate::{
        ConsistencyReport, Event, History, Key, KeyValuePair, Value, WeakestViolationReport,
    };

    fn parse_test_history(contents: &str) -> History {
        let mut sessions = Vec::new();
        for session in contents.split('=') {
            let session = session.trim();
            if session.is_empty() {
                continue;
            }

            let mut transactions = Vec::new();
            for transaction in session.split('-') {
                let transaction = transaction.trim();
                if transaction.is_empty() {
                    continue;
                }

                let mut events = Vec::new();
                for event in transaction.lines() {
                    let event = event.trim();
                    if event.is_empty() {
                        continue;
                    }

                    let mut parts = event.split_whitespace();
                    let event_type = parts.next().unwrap();
                    let key = Key(parts.next().unwrap().parse().unwrap());
                    let value = Value(parts.next().unwrap().parse().unwrap());
                    let event = match event_type {
                        "r" => Event::Read(KeyValuePair { key, value }),
                        "w" => Event::Write(KeyValuePair { key, value }),
                        _ => panic!("Invalid event type"),
                    };
                    events.push(event);
                }
                transactions.push(crate::Transaction { events });
            }
            sessions.push(transactions);
        }
        History {
            sessions,
            aborted_writes: FxHashSet::default(),
        }
    }

    #[test_resources("res/tests/causal/**/*.txt")]
    fn test_causal(file: &str) {
        let contents = fs::read_to_string(file).unwrap();
        let history = parse_test_history(&contents);
        let mut checker = history.checker::<WeakestViolationReport>();
        assert!(checker.check_causal().is_success());
        assert!(checker.check_read_atomic().is_success());
        assert!(checker.check_read_committed().is_success());
    }

    #[test_resources("res/tests/read-atomic/**/*.txt")]
    fn test_read_atomic(file: &str) {
        let contents = fs::read_to_string(file).unwrap();
        let history = parse_test_history(&contents);
        let mut checker = history.checker::<WeakestViolationReport>();
        assert!(!checker.check_causal().is_success());
        assert!(checker.check_read_atomic().is_success());
        assert!(checker.check_read_committed().is_success());
    }

    #[test_resources("res/tests/read-committed/**/*.txt")]
    fn test_read_committed(file: &str) {
        let contents = fs::read_to_string(file).unwrap();
        let history = parse_test_history(&contents);
        let mut checker = history.checker::<WeakestViolationReport>();
        assert!(!checker.check_causal().is_success());
        assert!(!checker.check_read_atomic().is_success());
        assert!(checker.check_read_committed().is_success());
    }

    #[test_resources("res/tests/none/**/*.txt")]
    fn test_inconsistent(file: &str) {
        let contents = fs::read_to_string(file).unwrap();
        let history = parse_test_history(&contents);
        let mut checker = history.checker::<WeakestViolationReport>();
        assert!(!checker.check_causal().is_success());
        assert!(!checker.check_read_atomic().is_success());
        assert!(!checker.check_read_committed().is_success());
    }
}
