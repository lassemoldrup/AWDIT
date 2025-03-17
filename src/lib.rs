use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, VecDeque};
use std::fmt::{self, Display, Formatter, Write};

use checker::{ConsistencyViolation, HistoryChecker};
use clap::ValueEnum;
#[cfg(feature = "dbcop")]
use dbcop::db::history::HistParams;

use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use smallvec::SmallVec;
use util::intersect_map;
use vector_clock::VectorClock;

pub mod checker;
pub mod fenwick;
pub mod formats;
pub mod partial_order;
pub mod util;
pub mod vector_clock;

/// Represents a database history. Initial reads are handled explicitly, meaning
/// there should be a transaction writing 0 to all keys, if so desired.
pub struct History {
    pub sessions: Vec<Vec<Transaction>>,
    pub aborted_writes: FxHashSet<KeyValuePair>,
}

impl History {
    pub fn checker<F: FnMut(&ConsistencyViolation)>(
        &self,
        report_mode: ReportMode,
        on_violation: F,
    ) -> HistoryChecker<F> {
        HistoryChecker {
            history: self,
            on_violation,
            report_mode,
            any_violation: false,
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

#[derive(Serialize)]
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

/// A transaction of a database history.
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

/// An identifier for a transaction. Consists of the index of a session and the
/// index of the transaction inside that session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TransactionId(pub usize, pub usize);

impl TransactionId {
    pub fn pred(self) -> Option<Self> {
        if self.1 > 0 {
            Some(Self(self.0, self.1 - 1))
        } else {
            None
        }
    }
}

impl Display for TransactionId {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

/// An event (aka an operation) of a database history.
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

#[derive(Clone, ValueEnum, strum::Display, PartialEq, Eq, PartialOrd, Ord)]
#[strum(serialize_all = "kebab-case")]
pub enum ReportMode {
    /// Only report the first violation found.
    First,
    /// Report all read consistency violations, and otherwise stop on any
    /// violation.
    ReadConsistency,
    /// Report all read consistency violations and causal cycles. Stop on any
    /// other violation. Note that this incurs a slight performance penalty.
    CausalCycles,
    /// Report as many violations as possible. Note that this does not guarantee
    /// that the weakest violations present are reported.
    Full,
}
