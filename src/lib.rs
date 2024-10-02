use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::{self, Display, Formatter};
use std::{iter, mem};

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

    fn get_writes_per_key(&self) -> FxHashMap<Key, Vec<Vec<(usize, Value)>>> {
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
                    if let Some(&(t2_idx, _)) = session_writes.last() {
                        if t2_idx == t_idx {
                            // Multiple writes in the same transaction to the same location
                            continue;
                        }
                    }
                    session_writes.push((t_idx, kv.value));
                }
            }
        }
        writes
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
                self.report.add_violation($violation);
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
                self.report.add_violation($violation);
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
        let mut read_values: Vec<Vec<FxHashMap<Key, Value>>> = graph
            .reads
            .iter()
            .map(|s| vec![FxHashMap::default(); s.len()])
            .collect();
        for (s_idx, session) in graph.reads.iter().enumerate() {
            for (t_idx, reads) in session.iter().enumerate() {
                for &(writer_tid, kv) in reads {
                    let writers = read_map[s_idx][t_idx].entry(kv.key).or_default();
                    if writers.len() == 1 && writers[0] != writer_tid {
                        let violation = ConsistencyViolation::NonRepeatableRead {
                            reader_tid: TransactionId(s_idx, t_idx),
                            t1: writer_tid,
                            t2: writers[0],
                            r1: kv,
                            r2: KeyValuePair {
                                key: kv.key,
                                value: read_values[s_idx][t_idx][&kv.key],
                            },
                        };
                        self.report.add_violation(violation);
                        if !R::IS_EXHAUSTIVE {
                            return Err(violation);
                        }
                    }
                    writers.push(writer_tid);
                    read_values[s_idx][t_idx].insert(kv.key, kv.value);
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

        let (hb, result) = graph.compute_hb(!R::IS_EXHAUSTIVE);
        if let Err((t1, t2)) = result {
            self.report
                .add_violation(ConsistencyViolation::CyclicHb { t1, t2 });
            if !R::IS_EXHAUSTIVE {
                return mem::take(&mut self.report);
            }
        };

        let history = self.history;
        let writes_per_key = history.get_writes_per_key();

        let mut rev_commit_order: Vec<_> = history
            .sessions
            .iter()
            .map(|s| vec![Vec::new(); s.len()])
            .collect();
        for (t3_s_idx, sess_reads) in graph.reads.iter().enumerate() {
            let mut last_writes_per_key = FxHashMap::default();
            for (t3_t_idx, t3_reads) in sess_reads.iter().enumerate() {
                // TODO: Maybe this should be replaced with an index map?
                let all_writers: FxHashMap<_, _> = t3_reads.iter().copied().collect();
                let mut prev_writers = FxHashMap::default();
                for &(t1, kv) in t3_reads {
                    let x = kv.key;
                    let last_writes: &mut Vec<isize> = last_writes_per_key
                        .entry(x)
                        .or_insert_with(|| vec![-1; history.sessions.len()]);
                    for (t2_s_idx, last_write) in last_writes.iter_mut().enumerate() {
                        let Ok(last_pred) = usize::try_from(hb[t3_s_idx][t3_t_idx][t2_s_idx])
                        else {
                            // If -1, no predecessors in t2's session
                            continue;
                        };
                        // Find the last write to x in t2's session that is less than or equal to last_pred
                        let writes = &writes_per_key[&x][t2_s_idx];
                        // TOOD: Test binary search
                        for write_idx in 0.max(*last_write)..writes.len() as isize {
                            match writes[write_idx as usize].0.cmp(&last_pred) {
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
                            let (t2_t_idx, t2_value) = writes[*last_write as usize];
                            let t2 = TransactionId(t2_s_idx, t2_t_idx);
                            if t2 == t1 {
                                continue;
                            } else if hb[t2_s_idx][t2_t_idx][t1.0] >= t1.1 as isize {
                                let t3 = TransactionId(t3_s_idx, t3_t_idx);
                                let shadowed_write = KeyValuePair {
                                    key: kv.key,
                                    value: t2_value,
                                };
                                let violation = if let Some(&read_y) = prev_writers.get(&t2) {
                                    ConsistencyViolation::NonMonoReadHb {
                                        t1,
                                        t2,
                                        t3,
                                        shadowed_write,
                                        read_x: kv,
                                        read_y,
                                    }
                                } else if t2_s_idx == t3_s_idx && t2_t_idx < t3_t_idx {
                                    ConsistencyViolation::FracturedReadHbSo {
                                        t1,
                                        t2,
                                        t3,
                                        shadowed_write,
                                        read_x: kv,
                                    }
                                } else if let Some(&read_y) = all_writers.get(&t2) {
                                    ConsistencyViolation::FracturedReadHbWr {
                                        t1,
                                        t2,
                                        t3,
                                        shadowed_write,
                                        read_x: kv,
                                        read_y,
                                    }
                                } else {
                                    ConsistencyViolation::HbConflictCo {
                                        t1,
                                        t2,
                                        t3,
                                        shadowed_write,
                                        read_x: kv,
                                    }
                                };
                                self.report.add_violation(violation);
                                if !R::IS_EXHAUSTIVE {
                                    return mem::take(&mut self.report);
                                }
                            }
                            rev_commit_order[t1.0][t1.1].push(t2);
                        }
                    }
                    prev_writers.insert(t1, kv);
                }
            }
        }

        // Check for cycles in the reverse commit order
        let cycle = graph.dfs(
            |TransactionId(s_idx, t_idx)| rev_commit_order[s_idx][t_idx].iter().copied(),
            |_| {},
            false,
        );
        if let Err((t1, t2)) = cycle {
            self.report
                .add_violation(ConsistencyViolation::Placeholder { t1, t2 });
        }
        mem::take(&mut self.report)
    }

    pub fn check_read_atomic(&mut self) -> R {
        if self.check_intra_transactional().is_err() {
            return mem::take(&mut self.report);
        }
        let Ok(graph) = self.infer_graph() else {
            return mem::take(&mut self.report);
        };

        // TODO: Avoid double checking
        let cycle = graph.dfs(|_| iter::empty(), |_| {}, !R::IS_EXHAUSTIVE);
        if let Err((t1, t2)) = cycle {
            self.report
                .add_violation(ConsistencyViolation::CyclicHb { t1, t2 });
            if !R::IS_EXHAUSTIVE {
                return mem::take(&mut self.report);
            }
        }

        let Ok(repeatable_reads_graph) = self.get_repeatable_reads_graph(&graph) else {
            return mem::take(&mut self.report);
        };

        let history = self.history;
        let write_sets = history.get_write_sets();

        let mut rev_commit_order: Vec<_> = history
            .sessions
            .iter()
            .map(|s| vec![Vec::new(); s.len()])
            .collect();
        for (t3_s_idx, sess_reads) in graph.reads.iter().enumerate() {
            let mut last_writes_per_key = FxHashMap::default();
            for (t3_t_idx, t3_writers) in sess_reads.iter().enumerate() {
                let t3_writes = &write_sets[t3_s_idx][t3_t_idx];
                for &(t1, kv) in t3_writers {
                    if let Some(&t2) = last_writes_per_key.get(&kv.key) {
                        if t2 == t1 {
                            continue;
                        }
                        rev_commit_order[t1.0][t1.1].push(t2);
                    }
                }
                let mut t3_writers = t3_writers.iter().map(|(t2, _)| *t2).collect_vec();
                t3_writers.sort();
                for t2 in t3_writers.into_iter().dedup() {
                    for &t1 in intersect_map(
                        &repeatable_reads_graph[t3_s_idx][t3_t_idx],
                        &write_sets[t2.0][t2.1],
                    )
                    .flatten()
                    .filter(|&&t1| t1 != t2)
                    {
                        rev_commit_order[t1.0][t1.1].push(t2);
                    }
                }
                for &k in t3_writes {
                    last_writes_per_key
                        .entry(k)
                        .or_insert(TransactionId(t3_s_idx, t3_t_idx));
                }
            }
        }

        // Check for cycles in the reverse commit order
        let cycle = graph.dfs(
            |TransactionId(s_idx, t_idx)| rev_commit_order[s_idx][t_idx].iter().copied(),
            |_| {},
            false,
        );
        if let Err((t1, t2)) = cycle {
            self.report
                .add_violation(ConsistencyViolation::Placeholder { t1, t2 });
        }
        mem::take(&mut self.report)
    }

    pub fn check_read_committed(&mut self) -> R {
        if self.check_intra_transactional().is_err() {
            return mem::take(&mut self.report);
        }
        let Ok(graph) = self.infer_graph() else {
            return mem::take(&mut self.report);
        };

        // TODO: Avoid double checking
        let cycle = graph.dfs(|_| iter::empty(), |_| {}, !R::IS_EXHAUSTIVE);
        if let Err((t1, t2)) = cycle {
            self.report
                .add_violation(ConsistencyViolation::CyclicHb { t1, t2 });
            if !R::IS_EXHAUSTIVE {
                return mem::take(&mut self.report);
            }
        }

        let history = self.history;
        let write_sets = history.get_write_sets();

        let mut rev_commit_order: Vec<_> = history
            .sessions
            .iter()
            .map(|s| vec![Vec::new(); s.len()])
            .collect();
        for t3_writers in graph.reads.iter().flatten() {
            let mut earliest_writer_per_loc: FxHashMap<Key, TransactionId> = FxHashMap::default();
            for &(t2, kv) in t3_writers.iter().rev() {
                for &t1 in intersect_map(&earliest_writer_per_loc, &write_sets[t2.0][t2.1]) {
                    if t1 != t2 {
                        rev_commit_order[t1.0][t1.1].push(t2);
                    }
                }
                earliest_writer_per_loc.insert(kv.key, t2);
            }
        }

        // Check for cycles in the reverse commit order
        let cycle = graph.dfs(
            |TransactionId(s_idx, t_idx)| rev_commit_order[s_idx][t_idx].iter().copied(),
            |_| {},
            false,
        );
        if let Err((t1, t2)) = cycle {
            self.report
                .add_violation(ConsistencyViolation::Placeholder { t1, t2 });
        }
        mem::take(&mut self.report)
    }
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
    rev_order: Vec<Vec<Vec<CommitOrderEdge>>>,
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
    ) {
        self.rev_order[t1.0][t1.1].push(CommitOrderEdge { t2, t3, x });
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
            tarjan_visit(
                TransactionId(s_idx, t_idx),
                graph,
                &mut state,
                &mut next_index,
                &mut stack,
            );
        }
    }
}

fn tarjan_visit(
    tid: TransactionId,
    graph: &WriteReadGraph,
    state: &mut Vec<Vec<TarjanState>>,
    next_index: &mut usize,
    stack: &mut Vec<TransactionId>,
) {
    let node_state = &mut state[tid.0][tid.1];
    node_state.index = *next_index;
    node_state.low_link = *next_index;
    *next_index += 1;
    stack.push(tid);
    node_state.on_stack = true;

    let index = node_state.index;
    let mut low_link = node_state.low_link;

    for tid2 in graph.rev_hb_edges(tid) {
        let tid2_node_state = state[tid2.0][tid2.1];
        if tid2_node_state.index == usize::MAX {
            tarjan_visit(tid2, graph, state, next_index, stack);
            low_link = low_link.min(state[tid2.0][tid2.1].low_link);
        } else if tid2_node_state.on_stack {
            low_link = low_link.min(tid2_node_state.low_link);
        }
    }

    state[tid.0][tid.1].low_link = low_link;
    if index == low_link {
        // TODO: pop from stack until tid
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

#[derive(Clone, Copy)]
struct CommitOrderEdge {
    t2: TransactionId,
    t3: TransactionId,
    x: KeyValuePair,
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

#[derive(Clone)]
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

type RepeatableReadsGraph = Vec<Vec<FxHashMap<Key, SmallVec<[TransactionId; 1]>>>>;

#[derive(Debug)]
struct WriteReadGraph {
    reads: Vec<Vec<Vec<(TransactionId, KeyValuePair)>>>,
}

impl WriteReadGraph {
    fn compute_hb(
        &self,
        stop_on_cycle: bool,
    ) -> (
        Vec<Vec<VectorClock>>,
        Result<(), (TransactionId, TransactionId)>,
    ) {
        let mut hb: Vec<Vec<VectorClock>> = self
            .reads
            .iter()
            .map(|s| vec![VectorClock::new_min(self.reads.len()); s.len()])
            .collect();

        let acyclic = self.dfs(
            |_| iter::empty(),
            |TransactionId(s_idx, t_idx)| {
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
            },
            stop_on_cycle,
        );

        (hb, acyclic)
    }

    fn dfs<I>(
        &self,
        additional_rev_edges: impl Fn(TransactionId) -> I,
        mut post_action: impl FnMut(TransactionId),
        stop_on_cycle: bool,
    ) -> Result<(), (TransactionId, TransactionId)>
    where
        I: IntoIterator<Item = TransactionId>,
    {
        let mut result = Ok(());
        let mut search_state: Vec<_> = self
            .reads
            .iter()
            .map(|s| vec![SearchState::NotSeen; s.len()])
            .collect();
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
                        if search_state[s_idx][t_idx] == SearchState::Marked {
                            // There were two parallel edges to this node
                            continue;
                        }
                        search_state[s_idx][t_idx] = SearchState::Marked;
                        stack.push(DfsStackEntry::Post(tid));

                        for TransactionId(s_idx, t_idx) in
                            self.rev_hb_edges(tid).chain(additional_rev_edges(tid))
                        {
                            match search_state[s_idx][t_idx] {
                                SearchState::NotSeen => {
                                    stack.push(DfsStackEntry::Pre(TransactionId(s_idx, t_idx)));
                                }
                                SearchState::Marked => {
                                    // Cycle detected
                                    let cycle = Err((tid, TransactionId(s_idx, t_idx)));
                                    if stop_on_cycle {
                                        return cycle;
                                    } else {
                                        result = cycle;
                                    }
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

        result
    }

    fn rev_hb_edges(
        &self,
        TransactionId(s_idx, t_idx): TransactionId,
    ) -> impl Iterator<Item = TransactionId> + Captures<'_> {
        let session_pred = t_idx
            .checked_sub(1)
            .map(|t_idx| TransactionId(s_idx, t_idx))
            .into_iter();
        let reads = self.reads[s_idx][t_idx].iter().map(|(writer, _)| *writer);
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
        match self.0 {
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

#[derive(thiserror::Error, Debug, Clone, Copy)]
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
    #[error("Transactions {t1} and {t2} causally depend on each other")]
    CyclicHb {
        t1: TransactionId,
        t2: TransactionId,
    },
    #[error("Non-monotonic read (HB): Transaction {t1} causes {t2}, and {t3} reads {read_x} from {t1} instead of {shadowed_write} from {t2}, from which it had read {read_y} earlier")]
    NonMonoReadHb {
        t1: TransactionId,
        t2: TransactionId,
        t3: TransactionId,
        shadowed_write: KeyValuePair,
        read_x: KeyValuePair,
        read_y: KeyValuePair,
    },
    #[error("Non-monotonic read (CO): Transaction {t1} commits before {t2}, and {t3} reads {read_x} from {t1} instead of {shadowed_write} from {t2}, from which it had read {read_y} earlier")]
    NonMonoReadCo {
        t1: TransactionId,
        t2: TransactionId,
        t3: TransactionId,
        shadowed_write: KeyValuePair,
        read_x: KeyValuePair,
        read_y: KeyValuePair,
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
    #[error("Fractured read (HB): Transaction {t1} causes {t2}, and {t3} reads {read_x} from {t1} instead of {shadowed_write} from {t2}, which is in the same session")]
    FracturedReadHbSo {
        t1: TransactionId,
        t2: TransactionId,
        t3: TransactionId,
        shadowed_write: KeyValuePair,
        read_x: KeyValuePair,
    },
    #[error("Fractured read (HB): Transaction {t1} causes {t2}, and {t3} reads {read_x} from {t1} instead of {shadowed_write} from {t2}, from which it reads {read_y}")]
    FracturedReadHbWr {
        t1: TransactionId,
        t2: TransactionId,
        t3: TransactionId,
        shadowed_write: KeyValuePair,
        read_x: KeyValuePair,
        read_y: KeyValuePair,
    },
    #[error("Fractured read (CO): Transaction {t1} commits before {t2}, and {t3} reads {read_x} from {t1} instead of {shadowed_write} from {t2}, which is in the same session")]
    FracturedReadCoSo {
        t1: TransactionId,
        t2: TransactionId,
        t3: TransactionId,
        shadowed_write: KeyValuePair,
        read_x: KeyValuePair,
    },
    #[error("Fractured read (CO): Transaction {t1} commits before {t2}, and {t3} reads {read_x} from {t1} instead of {shadowed_write} from {t2}, from which it reads {read_y}")]
    FracturedReadCoWr {
        t1: TransactionId,
        t2: TransactionId,
        t3: TransactionId,
        shadowed_write: KeyValuePair,
        read_x: KeyValuePair,
        read_y: KeyValuePair,
    },
    #[error("Transaction {t1} causes {t2}, and {t3} reads {read_x} from {t1} instead of {shadowed_write} from its causal predecessor {t2}")]
    HbConflictCo {
        t1: TransactionId,
        t2: TransactionId,
        t3: TransactionId,
        shadowed_write: KeyValuePair,
        read_x: KeyValuePair,
    },
    #[error("Transaction {t1} commits before {t2}, and {t3} reads {read_x} from {t1} instead of {shadowed_write} from its causal predecessor {t2}")]
    ConflictCo {
        t1: TransactionId,
        t2: TransactionId,
        t3: TransactionId,
        shadowed_write: KeyValuePair,
        read_x: KeyValuePair,
    },
    #[error("Cycle in CO: {t1} <-> {t2}")]
    Placeholder {
        t1: TransactionId,
        t2: TransactionId,
    },
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
