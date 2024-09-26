use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::iter;

use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};
use util::GetTwoMut;
use vector_clock::VectorClock;

pub mod fenwick;
pub mod partial_order;
pub mod util;
pub mod vector_clock;

pub struct History {
    pub sessions: Vec<Vec<Transaction>>,
}

impl History {
    fn infer_graph(&self) -> WriteReadGraph {
        let mut value_map = FxHashMap::default();
        for (s_idx, session) in self.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                for event in &transaction.events {
                    if let &Event::Write(key, value) = event {
                        let already_present =
                            value_map.insert((key, value), TransactionId(s_idx, t_idx));
                        assert!(already_present.is_none());
                    }
                }
            }
        }

        let mut graph = WriteReadGraph {
            reads: self
                .sessions
                .iter()
                .map(|s| vec![Vec::new(); s.len()])
                .collect(),
        };
        for (s_idx, session) in self.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                for event in &transaction.events {
                    if let &Event::Read(key, value) = event {
                        let writer = *value_map
                            .get(&(key, value))
                            .expect("Read did not have a matching write");
                        graph.reads[s_idx][t_idx].push((writer, key));
                    }
                }
            }
        }
        graph
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
                        Event::Read(..) => {}
                        Event::Write(k, _) => {
                            writes.insert(k);
                        }
                    }
                }
            }
        }
        write_sets
    }

    fn get_read_sets(&self) -> Vec<Vec<FxHashSet<Key>>> {
        let mut read_sets: Vec<Vec<FxHashSet<_>>> = self
            .sessions
            .iter()
            .map(|sess| vec![FxHashSet::default(); sess.len()])
            .collect();
        for (s_idx, session) in self.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                let reads = &mut read_sets[s_idx][t_idx];
                for &e in &transaction.events {
                    match e {
                        Event::Read(k, _) => {
                            reads.insert(k);
                        }
                        Event::Write(..) => {}
                    }
                }
            }
        }
        read_sets
    }

    fn get_writes_per_key(&self) -> FxHashMap<Key, Vec<Vec<usize>>> {
        let mut writes = FxHashMap::default();
        for (s_idx, session) in self.sessions.iter().enumerate() {
            for (t_idx, transaction) in session.iter().enumerate() {
                for event in &transaction.events {
                    let &Event::Write(key, _) = event else {
                        continue;
                    };
                    let session = &mut writes
                        .entry(key)
                        .or_insert_with(|| vec![Vec::new(); self.sessions.len()])[s_idx];
                    if session.last() == Some(&t_idx) {
                        // Multiple writes in the same transaction to the same location
                        continue;
                    }
                    session.push(t_idx);
                }
            }
        }
        writes
    }

    pub fn check_causal(&self) -> bool {
        let graph = self.infer_graph();
        let Some(hb) = graph.compute_hb() else {
            eprintln!("HB cycle detected");
            return false;
        };
        let writes_per_key = self.get_writes_per_key();

        let mut rev_commit_order: Vec<_> = self
            .sessions
            .iter()
            .map(|s| vec![Vec::new(); s.len()])
            .collect();
        for (t3_s_idx, sess_reads) in graph.reads.iter().enumerate() {
            let mut last_writes_per_key = FxHashMap::default();
            for (t3_t_idx, t3_reads) in sess_reads.iter().enumerate() {
                for &(t1, x) in t3_reads {
                    let last_writes: &mut Vec<isize> = last_writes_per_key
                        .entry(x)
                        .or_insert_with(|| vec![-1; self.sessions.len()]);
                    for (t2_s_idx, last_write) in last_writes.iter_mut().enumerate() {
                        let Ok(last_pred) = usize::try_from(hb[t3_s_idx][t3_t_idx][t2_s_idx])
                        else {
                            // If -1, no predecessors in t2_s_idx
                            continue;
                        };
                        // Find the last write to x in t2's session that is less than or equal to last_pred
                        let writes = &writes_per_key[&x][t2_s_idx];
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
                            let t2 = TransactionId(t2_s_idx, writes[*last_write as usize]);
                            if t2 == t1 {
                                continue;
                            } else if t2.0 == t1.0 && t2.1 > t1.1 {
                                eprintln!("Cycle detected: {:?} <-> {:?}", t1, t2);
                                return false;
                            }
                            rev_commit_order[t1.0][t1.1]
                                .push(TransactionId(t2_s_idx, writes[*last_write as usize]));
                        }
                    }
                }
            }
        }

        // Check for cycles in the reverse commit order
        graph.dfs(
            |TransactionId(s_idx, t_idx)| rev_commit_order[s_idx][t_idx].iter().copied(),
            |_| {},
        )
    }

    pub fn check_read_atomic(&self) -> bool {
        let graph = self.infer_graph();
        let Ok(repeatable_reads_graph) = graph.to_repeatable_reads_graph() else {
            return false;
        };
        let write_sets = self.get_write_sets();
        let read_sets = self.get_read_sets();

        let mut rev_commit_order: Vec<_> = self
            .sessions
            .iter()
            .map(|s| vec![Vec::new(); s.len()])
            .collect();
        for (t3_s_idx, sess_reads) in graph.reads.iter().enumerate() {
            let mut last_writes_per_key = FxHashMap::default();
            for (t3_t_idx, t3_writers) in sess_reads.iter().enumerate() {
                let t3_writes = &write_sets[t3_s_idx][t3_t_idx];
                let t3_reads = &read_sets[t3_s_idx][t3_t_idx];
                for &(t1, k) in t3_writers {
                    if let Some(&t2) = last_writes_per_key.get(&k) {
                        if t2 != t1 {
                            rev_commit_order[t1.0][t1.1].push(t2);
                        }
                    }
                }
                let mut t3_writers = t3_writers.iter().map(|(t2, _)| *t2).collect_vec();
                t3_writers.sort();
                for t2 in t3_writers.into_iter().dedup() {
                    for k in write_sets[t2.0][t2.1].intersection(t3_reads) {
                        let t1 = repeatable_reads_graph[t3_s_idx][t3_t_idx][k];
                        if t1 != t2 {
                            rev_commit_order[t1.0][t1.1].push(t2);
                        }
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
        graph.dfs(
            |TransactionId(s_idx, t_idx)| rev_commit_order[s_idx][t_idx].iter().copied(),
            |_| {},
        )
    }

    pub fn check_read_committed(&self) -> bool {
        let graph = self.infer_graph();
        let write_sets = self.get_write_sets();

        let mut rev_commit_order: Vec<_> = self
            .sessions
            .iter()
            .map(|s| vec![Vec::new(); s.len()])
            .collect();
        for t3_writers in graph.reads.iter().flatten() {
            let mut earliest_writer_per_loc = FxHashMap::default();
            let mut reads = FxHashSet::default();
            for &(t2, k1) in t3_writers.iter().rev() {
                for k2 in write_sets[t2.0][t2.1].intersection(&reads) {
                    let t1: TransactionId = earliest_writer_per_loc[k2];
                    if t1 != t2 {
                        rev_commit_order[t1.0][t1.1].push(t2);
                    }
                }
                earliest_writer_per_loc.insert(k1, t2);
                reads.insert(k1);
            }
        }

        // Check for cycles in the reverse commit order
        graph.dfs(
            |TransactionId(s_idx, t_idx)| rev_commit_order[s_idx][t_idx].iter().copied(),
            |_| {},
        )
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

#[derive(Clone, Copy, strum::Display)]
pub enum Event {
    #[strum(serialize = "r {0} {1}")]
    Read(Key, Value),
    #[strum(serialize = "w {0} {1}")]
    Write(Key, Value),
}

impl Event {
    pub fn key(&self) -> Key {
        match self {
            Event::Read(key, _) | Event::Write(key, _) => *key,
        }
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

type RepeatableReadsGraph = Vec<Vec<FxHashMap<Key, TransactionId>>>;

#[derive(Debug)]
struct WriteReadGraph {
    reads: Vec<Vec<Vec<(TransactionId, Key)>>>,
}

impl WriteReadGraph {
    fn compute_hb(&self) -> Option<Vec<Vec<VectorClock>>> {
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
        );

        acyclic.then_some(hb)
    }

    fn dfs<I>(
        &self,
        additional_rev_edges: impl Fn(TransactionId) -> I,
        mut post_action: impl FnMut(TransactionId),
    ) -> bool
    where
        I: IntoIterator<Item = TransactionId>,
    {
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

                        let session_pred = t_idx
                            .checked_sub(1)
                            .map(|t_idx| TransactionId(s_idx, t_idx))
                            .into_iter();
                        let reads = self.reads[s_idx][t_idx].iter().map(|(writer, _)| *writer);

                        for TransactionId(s_idx, t_idx) in
                            session_pred.chain(reads).chain(additional_rev_edges(tid))
                        {
                            match search_state[s_idx][t_idx] {
                                SearchState::NotSeen => {
                                    stack.push(DfsStackEntry::Pre(TransactionId(s_idx, t_idx)));
                                }
                                SearchState::Marked => {
                                    // Cycle detected
                                    eprintln!(
                                        "Cycle in DFS: {tid:?} <-> {:?}",
                                        TransactionId(s_idx, t_idx)
                                    );
                                    return false;
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

        true
    }

    /// Fails if repeatable reads does not hold
    fn to_repeatable_reads_graph(&self) -> Result<RepeatableReadsGraph, ()> {
        let mut read_map: RepeatableReadsGraph = self
            .reads
            .iter()
            .map(|s| vec![FxHashMap::default(); s.len()])
            .collect();
        for (s_idx, session) in self.reads.iter().enumerate() {
            for (t_idx, reads) in session.iter().enumerate() {
                for &(writer_tid, k) in reads {
                    if matches!(read_map[s_idx][t_idx].insert(k, writer_tid), Some(tid) if tid != writer_tid)
                    {
                        eprintln!("Non repeatable read: ({s_idx}, {t_idx})");
                        return Err(());
                    }
                }
            }
        }

        Ok(read_map)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use test_generator::test_resources;

    use crate::{Event, History, Key, Value};

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
                    let key = parts.next().unwrap().parse().unwrap();
                    let value = parts.next().unwrap().parse().unwrap();
                    let event = match event_type {
                        "r" => Event::Read(Key(key), Value(value)),
                        "w" => Event::Write(Key(key), Value(value)),
                        _ => panic!("Invalid event type"),
                    };
                    events.push(event);
                }
                transactions.push(crate::Transaction { events });
            }
            sessions.push(transactions);
        }
        History { sessions }
    }

    #[test_resources("res/tests/causal/**/*.txt")]
    fn test_causal(file: &str) {
        let contents = fs::read_to_string(file).unwrap();
        let history = parse_test_history(&contents);
        assert!(history.check_causal());
        assert!(history.check_read_atomic());
        assert!(history.check_read_committed());
    }

    #[test_resources("res/tests/read-atomic/**/*.txt")]
    fn test_read_atomic(file: &str) {
        let contents = fs::read_to_string(file).unwrap();
        let history = parse_test_history(&contents);
        assert!(!history.check_causal());
        assert!(history.check_read_atomic());
        assert!(history.check_read_committed());
    }

    #[test_resources("res/tests/read-committed/**/*.txt")]
    fn test_read_committed(file: &str) {
        let contents = fs::read_to_string(file).unwrap();
        let history = parse_test_history(&contents);
        assert!(!history.check_causal());
        assert!(!history.check_read_atomic());
        assert!(history.check_read_committed());
    }

    #[test_resources("res/tests/none/**/*.txt")]
    fn test_inconsistent(file: &str) {
        let contents = fs::read_to_string(file).unwrap();
        let history = parse_test_history(&contents);
        assert!(!history.check_causal());
        assert!(!history.check_read_atomic());
        assert!(!history.check_read_committed());
    }
}
