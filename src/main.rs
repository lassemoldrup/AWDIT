use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

use anyhow::Context;
use awdit::checker::{ConsistencyReport, FullViolationReport, WeakestViolationReport};
use awdit::util::{intersect_map, GetTwoMut};
use awdit::vector_clock::VectorClock;
use awdit::{Event, History, Key, KeyValuePair, Transaction, TransactionId, Value};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rand::prelude::*;
use rand_distr::{Bernoulli, Pareto, Uniform};
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Parser)]
struct App {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a history with the given parameters.
    Generate(GenerateArgs),
    /// Check a history for consistency violations.
    Check {
        #[clap(short, long, default_value_t = IsolationLevel::Causal)]
        isolation: IsolationLevel,
        #[arg(required = true)]
        path: PathBuf,
        #[clap(short, long, default_value_t = HistoryFormat::Plume)]
        format: HistoryFormat,
        #[clap(short, long, default_value_t = ReportMode::Full)]
        report_mode: ReportMode,
        /// The size of the stack in MiB, can be increased if stack overflows occur.
        #[clap(short, long, default_value_t = 32)]
        stack_size: usize,
    },
    /// Convert a Plume/Cobra/DBCop history to a Plume/DBCop/Test history.
    Convert {
        #[arg(required = true)]
        from_path: PathBuf,
        #[arg(required = true)]
        to_path: PathBuf,
        #[clap(short, long, default_value_t = false)]
        /// Zeros the MSB of each key and value (needed because Plume
        /// only supports signed 64-bit numbers).
        max_63_bits: bool,
        #[clap(short = 'F', long, default_value_t = false)]
        /// "Fix" thin-air reads by changing them to reads of the
        /// initial value (not necessarily sound).
        fix_thin_air_reads: bool,
        #[clap(short, long, default_value_t = HistoryFormat::Cobra)]
        from_format: HistoryFormat,
        #[clap(short, long, default_value_t = HistoryFormat::Plume)]
        to_format: HistoryFormat,
    },
    /// Print statistics about a history.
    Stats {
        #[arg(required = true)]
        path: PathBuf,
        #[clap(short, long, default_value_t = HistoryFormat::Plume)]
        format: HistoryFormat,
        #[clap(short, long = "json", default_value_t = false)]
        json_output: bool,
    },
}

#[derive(Args)]
struct GenerateArgs {
    #[clap(short, long, default_value_t = 8)]
    locations: usize,
    #[clap(short, long, default_value_t = 20)]
    events: usize,
    #[clap(short, long, default_value_t = IsolationLevel::Causal)]
    isolation: IsolationLevel,
    #[clap(short, long, default_value_t = HistoryFormat::Plume)]
    format: HistoryFormat,
    #[arg(required = true)]
    path: PathBuf,
    #[clap(short, long, default_value_t = 0.8)]
    read_ratio: f64,
    #[clap(short, long, default_value_t = 5.)]
    mean_transaction_size: f64,
    #[clap(short, long, default_value_t = 1.16)]
    alpha: f64,
}

impl GenerateArgs {
    fn _find_inconsistent(&self) {
        for _ in 0..100000 {
            let history = PartialHistory::generate(self).into_read_committed_history();
            let mut checker = history.checker::<WeakestViolationReport>();
            if !checker.check_read_atomic().is_success() {
                println!("{history}");
                return;
            }
        }
        println!("Failed to find inconsistent")
    }
}

#[derive(Clone, PartialEq, Eq, ValueEnum, strum::Display)]
#[strum(serialize_all = "kebab-case")]
enum IsolationLevel {
    ReadCommitted,
    ReadAtomic,
    Causal,
}

#[derive(Clone, ValueEnum, strum::EnumString, strum::Display, PartialEq, Eq)]
#[strum(serialize_all = "kebab-case")]
enum HistoryFormat {
    Plume,
    Cobra,
    #[cfg(feature = "dbcop")]
    #[strum(serialize = "dbcop")]
    DbCop,
    Test,
}

#[derive(Clone, ValueEnum, strum::Display)]
#[strum(serialize_all = "kebab-case")]
enum ReportMode {
    /// Only report the first violation found.
    First,
    /// Stop after reporting all read consistency violations (continues if none are found).
    ReadConsistency,
    /// Stop on causal cycles (continues if none are found).
    CausalCycles,
    /// Report as many violations as possible. Note that this does not guarantee that the
    /// weakest violations present are reported.
    Full,
}

struct PartialHistory {
    sessions: Vec<Vec<Transaction>>,
    commit_order: Vec<TransactionId>,
    rev_commit_order: FxHashMap<TransactionId, usize>,
    locations: usize,
}

impl PartialHistory {
    fn generate(app: &GenerateArgs) -> Self {
        if app.alpha <= 0. {
            panic!("α must be positive");
        }

        let num_sessions = (app.events as f64).sqrt().ceil() as usize;
        if num_sessions == 0 {
            panic!("Number of sessions must be at least 1");
        }
        // The last session is contains initial values for all keys
        let mut sessions = vec![vec![Transaction::new()]; num_sessions + 1];

        // Generate initial transaction
        sessions[num_sessions][0]
            .events
            .extend((0..app.locations).map(|k| {
                Event::Write(KeyValuePair {
                    key: Key(k),
                    value: Value(0),
                })
            }));

        let session_dist = Uniform::new(0, num_sessions);
        let event_dist = Bernoulli::new(app.read_ratio).unwrap();
        let commit_dist = Bernoulli::new(1. / app.mean_transaction_size).unwrap();

        let mut rng = rand::thread_rng();
        let mut key_shuffle = (0..app.locations).collect::<Vec<_>>();
        key_shuffle.shuffle(&mut rng);
        // let alpha = 0.2f64.log2() / (5f64.log2() - (app.locations as f64).log2());
        // let min_x = 1.;
        let min_x = (0.2f64).powf(1. / app.alpha) * (app.locations as f64 / 5.)
            / (1. - (0.2f64).powf(1. / app.alpha));
        let key_dist = Pareto::new(min_x, app.alpha).unwrap().map(move |k| {
            let k = (k - min_x).round() as usize;
            let key = if k < app.locations {
                key_shuffle[k]
            } else {
                rand::thread_rng().gen_range(0..app.locations)
            };
            Key(key)
        });

        let mut value_map = FxHashMap::default();

        let mut commit_order = vec![TransactionId(num_sessions, 0)];
        let mut rev_commit_order: FxHashMap<_, _> =
            [(TransactionId(num_sessions, 0), 0)].into_iter().collect();
        for _ in 0..app.events {
            let session = session_dist.sample(&mut rng);
            let mut tid = TransactionId(session, sessions[session].len() - 1);

            let key = key_dist.sample(&mut rng);
            let mut should_commit = false;
            for e in sessions[session].last().unwrap().events.iter() {
                match e {
                    // If key already written to, commit the transaction
                    Event::Write(kv) if kv.key == key => should_commit = true,
                    _ => {}
                }
            }
            if should_commit {
                commit_order.push(tid);
                rev_commit_order.insert(tid, commit_order.len() - 1);
                sessions[session].push(Transaction::new());
                tid = TransactionId(tid.0, tid.1 + 1);
            }

            let is_read = event_dist.sample(&mut rng);
            let event = if is_read {
                // The value will be updated after the commit order is determined
                Event::Read(KeyValuePair {
                    key,
                    value: Value(0),
                })
            } else {
                let value = Value(*value_map.entry(key).and_modify(|v| *v += 1).or_insert(1));
                Event::Write(KeyValuePair { key, value })
            };
            sessions[session].last_mut().unwrap().push(event);

            let should_commit = commit_dist.sample(&mut rand::thread_rng());
            if should_commit {
                commit_order.push(tid);
                rev_commit_order.insert(tid, commit_order.len() - 1);
                sessions[session].push(Transaction::new());
            }
        }

        // Remove empty transactions and commit other entries
        for (s_idx, session) in sessions[..num_sessions].iter_mut().enumerate() {
            if session.last().unwrap().events.is_empty() {
                session.pop();
            } else {
                let tid = TransactionId(s_idx, session.len() - 1);
                commit_order.push(tid);
                rev_commit_order.insert(tid, commit_order.len() - 1);
            }
        }

        Self {
            sessions,
            commit_order,
            rev_commit_order,
            locations: app.locations,
        }
    }

    fn into_causal_history(mut self) -> History {
        let mut rng = rand::thread_rng();
        let non_init_sessions = self.sessions.len() - 1;

        // Update read events
        let mut hb = self
            .sessions
            .iter()
            .map(|s| vec![VectorClock::new_min(self.sessions.len()); s.len()])
            .collect::<Vec<_>>();
        let mut writes_per_loc_per_session: Vec<FxHashMap<Key, Vec<usize>>> =
            vec![FxHashMap::default(); self.sessions.len()];
        // Values written by transactions in commit order
        let mut writes_per_loc: Vec<Vec<(TransactionId, Value)>> = vec![vec![]; self.locations];
        let mut idx_in_writes_per_loc = vec![FxHashMap::default(); self.locations];
        for tid @ &TransactionId(s_idx, t_idx) in self.commit_order.iter() {
            if t_idx > 0 {
                let (pred, succ) = hb[s_idx].get_two_mut(t_idx - 1, t_idx);
                succ.join(pred);
                succ.join1(s_idx, t_idx as isize - 1);
            }
            let mut read_map = FxHashMap::default();
            let mut writers: BTreeMap<usize, Vec<Key>> = BTreeMap::new();
            for event in self.sessions[s_idx][t_idx].events.iter_mut() {
                match event {
                    Event::Read(kv) => {
                        // We implement repeatable reads explicitly as an optimization
                        if let Some(&(_, write_value)) = read_map.get(&kv.key) {
                            kv.value = write_value;
                            continue;
                        }
                        // Find co-first write that the read is allowed to read from
                        let earliest_legal_write = hb[s_idx][t_idx]
                            .iter()
                            .enumerate()
                            .filter_map(pair_to_tid)
                            .filter_map(|tid| {
                                // Last write to k in the session
                                writes_per_loc_per_session[tid.0]
                                    .get(&kv.key)
                                    .and_then(|ws| {
                                        ws.partition_point(|&i| i <= tid.1)
                                            .checked_sub(1)
                                            .map(|i| TransactionId(tid.0, ws[i]))
                                    })
                            })
                            .max_by_key(|tid| self.rev_commit_order[tid])
                            .unwrap_or(TransactionId(non_init_sessions, 0));
                        let earliest_legal_idx =
                            idx_in_writes_per_loc[kv.key.0][&earliest_legal_write];

                        // Use rejection sampling to determine a choice out of the remaining possibilities
                        let num_choices = writes_per_loc[kv.key.0].len() - earliest_legal_idx;
                        let mut choices = (0..num_choices).collect::<Vec<_>>();
                        choices.shuffle(&mut rng);
                        let mut consistent_choice = None;
                        'choice_loop: for choice in choices {
                            let (write_tid, write_value) =
                                writes_per_loc[kv.key.0][earliest_legal_idx + choice];
                            let write_co_idx = self.rev_commit_order[&write_tid];

                            // Optimization: we know it's safe to read from writes co-before any of our other writers
                            // Optimization: if we are already reading from a transaction, we can do so again
                            let min_writer_co_idx = writers
                                .first_key_value()
                                .map(|(i, _)| *i)
                                .unwrap_or(usize::MAX);
                            if write_co_idx <= min_writer_co_idx
                                || writers.contains_key(&write_co_idx)
                            {
                                consistent_choice = Some((write_tid, write_value));
                                break;
                            }

                            // Other writers before the considered write could cause problems
                            for (&other_write_co_idx, keys) in writers.range(..write_co_idx) {
                                // If the considered write writes to any key of the same keys, there is a conflict
                                if keys
                                    .iter()
                                    .any(|k2| idx_in_writes_per_loc[k2.0].contains_key(&write_tid))
                                {
                                    continue 'choice_loop;
                                }
                                // Find the last write to k2 in each session before the considered write
                                // and check if it is after the other write
                                let inconsistent = hb[write_tid.0][write_tid.1]
                                    .iter()
                                    .enumerate()
                                    .filter_map(pair_to_tid)
                                    .flat_map(|tid| {
                                        // Saitsfy the compiler
                                        let writes_per_loc_per_session = &writes_per_loc_per_session;
                                        // Find the latest writer to each keys in keys
                                        keys.iter().filter_map(move |k2|
                                            // Last write to k2 in the session
                                            writes_per_loc_per_session[tid.0].get(k2).and_then(|ws| {
                                                ws.partition_point(|&i| i <= tid.1)
                                                    .checked_sub(1)
                                                    .map(|i| TransactionId(tid.0, ws[i]))
                                        }))
                                    })
                                    .any(|tid| self.rev_commit_order[&tid] > other_write_co_idx);
                                if inconsistent {
                                    continue 'choice_loop;
                                }
                            }

                            consistent_choice = Some((write_tid, write_value));
                        }

                        let (write_tid, write_value) =
                            consistent_choice.expect("Should have at least one valid writer");
                        kv.value = write_value;
                        writers
                            .entry(self.rev_commit_order[&write_tid])
                            .or_default()
                            .push(kv.key);
                        read_map.insert(kv.key, (write_tid, write_value));
                        hb[s_idx][t_idx].join1(write_tid.0, write_tid.1 as isize);
                        if write_tid.0 != s_idx {
                            let (pred_sess, succ_sess) = hb.get_two_mut(write_tid.0, s_idx);
                            succ_sess[t_idx].join(&pred_sess[write_tid.1]);
                        }
                    }
                    Event::Write(kv) => {
                        writes_per_loc_per_session[s_idx]
                            .entry(kv.key)
                            .or_default()
                            .push(t_idx);
                        idx_in_writes_per_loc[kv.key.0]
                            .insert(*tid, writes_per_loc[kv.key.0].len());
                        writes_per_loc[kv.key.0].push((*tid, kv.value));
                    }
                }
            }
        }

        self.sessions.retain(|s| !s.is_empty());
        self.sessions[..non_init_sessions].shuffle(&mut rng);
        History {
            sessions: self.sessions,
            aborted_writes: FxHashSet::default(),
        }
    }

    fn into_read_atomic_history(mut self) -> History {
        let mut rng = rand::thread_rng();
        let non_init_sessions = self.sessions.len() - 1;

        let mut writes_per_loc_per_session: Vec<FxHashMap<Key, Vec<usize>>> =
            vec![FxHashMap::default(); self.sessions.len()];
        // Values written by transactions in commit order
        let mut writes_per_loc: Vec<Vec<(TransactionId, Value)>> = vec![vec![]; self.locations];
        let mut idx_in_writes_per_loc = vec![FxHashMap::default(); self.locations];
        let mut write_sets: Vec<Vec<FxHashSet<Key>>> = self
            .sessions
            .iter()
            .map(|s| vec![FxHashSet::default(); s.len()])
            .collect();
        for tid @ &TransactionId(s_idx, t_idx) in self.commit_order.iter() {
            let mut read_map = FxHashMap::default();
            let mut writers = BTreeSet::new();
            for event in self.sessions[s_idx][t_idx].events.iter_mut() {
                match event {
                    Event::Read(kv) => {
                        // We implement repeatable reads explicitly
                        if let Some(&(_, write_value)) = read_map.get(&kv.key) {
                            kv.value = write_value;
                            continue;
                        }

                        // Find co-first write we are allowed to read from
                        let earliest_legal_write = writes_per_loc_per_session[s_idx]
                            .get(&kv.key)
                            // Search our own session for the last write to k before us
                            .and_then(|ws| {
                                ws.partition_point(|&i| i < t_idx)
                                    .checked_sub(1)
                                    .map(|i| TransactionId(s_idx, ws[i]))
                            })
                            .into_iter()
                            .chain(read_map.iter().map(|(_, (tid, _))| *tid))
                            .filter(|tid| write_sets[tid.0][tid.1].contains(&kv.key))
                            .max_by_key(|tid| self.rev_commit_order[tid])
                            .unwrap_or(TransactionId(non_init_sessions, 0));
                        let earliest_legal_idx =
                            idx_in_writes_per_loc[kv.key.0][&earliest_legal_write];

                        // Use rejection sampling to determine a choice out of the remaining possibilities
                        let num_choices = writes_per_loc[kv.key.0].len() - earliest_legal_idx;
                        let mut choices = (0..num_choices).collect::<Vec<_>>();
                        choices.shuffle(&mut rng);
                        let mut consistent_choice = None;
                        'choice_loop: for choice in choices {
                            let (write_tid, write_value) =
                                writes_per_loc[kv.key.0][earliest_legal_idx + choice];
                            let write_co_idx = self.rev_commit_order[&write_tid];

                            // Optimization: we know it's safe to read from writes co-before any of our other writers
                            // Optimization: if we are already reading from a transaction, we can do so again
                            let min_writer_co_idx = *writers.first().unwrap_or(&usize::MAX);
                            if write_co_idx <= min_writer_co_idx || writers.contains(&write_co_idx)
                            {
                                consistent_choice = Some((write_tid, write_value));
                                break;
                            }

                            // If there is a writer writing something in the considered write set,
                            // and that writer is earlier in the co order, that is inconsistent
                            for (_, (conflicting_tid, _)) in
                                intersect_map(&read_map, &write_sets[write_tid.0][write_tid.1])
                            {
                                if self.rev_commit_order[conflicting_tid] < write_co_idx {
                                    continue 'choice_loop;
                                }
                            }

                            consistent_choice = Some((write_tid, write_value));
                            break;
                        }

                        let (write_tid, write_value) =
                            consistent_choice.expect("Should have at least one valid writer");
                        let write_co_idx = self.rev_commit_order[&write_tid];
                        writers.insert(write_co_idx);
                        kv.value = write_value;
                        read_map.insert(kv.key, (write_tid, write_value));
                    }
                    Event::Write(kv) => {
                        writes_per_loc_per_session[s_idx]
                            .entry(kv.key)
                            .or_default()
                            .push(t_idx);
                        idx_in_writes_per_loc[kv.key.0]
                            .insert(*tid, writes_per_loc[kv.key.0].len());
                        writes_per_loc[kv.key.0].push((*tid, kv.value));
                        write_sets[s_idx][t_idx].insert(kv.key);
                    }
                }
            }
        }

        self.sessions.retain(|s| !s.is_empty());
        self.sessions[..non_init_sessions].shuffle(&mut rng);
        History {
            sessions: self.sessions,
            aborted_writes: FxHashSet::default(),
        }
    }

    fn into_read_committed_history(mut self) -> History {
        let mut rng = rand::thread_rng();
        let non_init_sessions = self.sessions.len() - 1;

        // Values written by transactions in commit order
        let mut writes_per_loc: Vec<Vec<(TransactionId, Value)>> = vec![vec![]; self.locations];
        let mut idx_in_writes_per_loc = vec![FxHashMap::default(); self.locations];
        let mut write_sets: Vec<Vec<FxHashSet<Key>>> = self
            .sessions
            .iter()
            .map(|s| vec![FxHashSet::default(); s.len()])
            .collect();
        for tid @ &TransactionId(s_idx, t_idx) in self.commit_order.iter() {
            let mut writers = FxHashSet::default();
            for event in self.sessions[s_idx][t_idx].events.iter_mut() {
                match event {
                    Event::Read(kv) => {
                        // Find co-first write we are allowed to read from
                        let &earliest_legal_write = writers
                            .iter()
                            .filter(|tid: &&TransactionId| {
                                write_sets[tid.0][tid.1].contains(&kv.key)
                            })
                            .max_by_key(|tid| self.rev_commit_order[tid])
                            .unwrap_or(&TransactionId(non_init_sessions, 0));
                        let earliest_legal_idx =
                            idx_in_writes_per_loc[kv.key.0][&earliest_legal_write];

                        let &(write_tid, write_value) = writes_per_loc[kv.key.0]
                            [earliest_legal_idx..]
                            .choose(&mut rng)
                            .expect("Should have at least one valid writer");
                        kv.value = write_value;
                        writers.insert(write_tid);
                    }
                    Event::Write(kv) => {
                        idx_in_writes_per_loc[kv.key.0]
                            .insert(*tid, writes_per_loc[kv.key.0].len());
                        writes_per_loc[kv.key.0].push((*tid, kv.value));
                        write_sets[s_idx][t_idx].insert(kv.key);
                    }
                }
            }
        }

        self.sessions.retain(|s| !s.is_empty());
        self.sessions[..non_init_sessions].shuffle(&mut rng);
        History {
            sessions: self.sessions,
            aborted_writes: FxHashSet::default(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let app = App::parse();
    match app.command {
        Command::Generate(args) => {
            // app._find_inconsistent();
            let history = PartialHistory::generate(&args);
            let mut history = match args.isolation {
                IsolationLevel::ReadCommitted => history.into_read_committed_history(),
                IsolationLevel::ReadAtomic => history.into_read_atomic_history(),
                IsolationLevel::Causal => history.into_causal_history(),
            };

            match args.format {
                HistoryFormat::Plume | HistoryFormat::DbCop => {
                    // Remove the init session
                    history.sessions.pop();
                }
                _ => {}
            }

            match args.format {
                HistoryFormat::Plume => history
                    .serialize_plume_history(args.path)
                    .context("Failed to write Plume history")?,
                HistoryFormat::Cobra => panic!("Cannot convert to Cobra history"),
                #[cfg(feature = "dbcop")]
                HistoryFormat::DbCop => history
                    .serialize_dbcop_history(args.path)
                    .context("Failed to write DBCop history")?,
                HistoryFormat::Test => history
                    .serialize_test_history(args.path)
                    .context("Failed to write test history")?,
            }
        }
        Command::Check {
            isolation,
            path,
            format,
            report_mode,
            stack_size,
        } => {
            let parsing_start = Instant::now();
            let history = match format {
                HistoryFormat::Plume => History::parse_plume_history(path)
                    .context("Failed to parse path as Plume history")?,
                HistoryFormat::Cobra => History::parse_cobra_history(path)
                    .context("Failed to parse path as Cobra history")?,
                #[cfg(feature = "dbcop")]
                HistoryFormat::DbCop => History::parse_dbcop_history(path)
                    .context("Failed to parse path as Cobra history")?,
                HistoryFormat::Test => History::parse_test_history(path)
                    .context("Failed to parse path as test history")?,
            };

            let parsing_elapsed = parsing_start.elapsed();
            println!("Done parsing: {}ms", parsing_elapsed.as_millis());

            macro_rules! check_history {
                ($report:ty) => {{
                    let mut checker = history.checker::<$report>();
                    match isolation {
                        IsolationLevel::ReadCommitted => checker.check_read_committed(),
                        IsolationLevel::ReadAtomic => checker.check_read_atomic(),
                        IsolationLevel::Causal => checker.check_causal(),
                    }
                }};
            }

            let checking_start = Instant::now();
            // Spawn a new thread to avoid stack overflow for big histories
            thread::scope(|scope| {
                thread::Builder::new()
                    .stack_size(stack_size * 1024 * 1024)
                    .spawn_scoped(scope, || todo!())
                    .unwrap();
            });
            let checking_elapsed = checking_start.elapsed();
            println!("Done checking: {}ms", checking_elapsed.as_millis());
        }
        Command::Convert {
            from_path,
            to_path,
            max_63_bits,
            fix_thin_air_reads,
            from_format,
            to_format,
        } => {
            let mut history = match from_format {
                HistoryFormat::Plume => History::parse_plume_history(&from_path)
                    .context("Failed to parse path as Plume history")?,
                HistoryFormat::Cobra => History::parse_cobra_history(&from_path)
                    .context("Failed to parse path as Cobra history")?,
                #[cfg(feature = "dbcop")]
                HistoryFormat::DbCop => History::parse_dbcop_history(&from_path)
                    .context("Failed to parse path as Cobra history")?,
                HistoryFormat::Test => panic!("Cannot convert from test history"),
            };
            match from_format {
                HistoryFormat::Plume | HistoryFormat::Cobra if to_format != HistoryFormat::Test => {
                    history.sessions.pop();
                }
                _ => {}
            }
            if max_63_bits {
                history.strip_64th_bit();
            }
            if fix_thin_air_reads {
                history.fix_thin_air_reads();
            }

            match to_format {
                HistoryFormat::Plume => history
                    .serialize_plume_history(&to_path)
                    .context("Failed to write Plume history")?,
                HistoryFormat::Cobra => panic!("Cannot convert to Cobra history"),
                #[cfg(feature = "dbcop")]
                HistoryFormat::DbCop => history
                    .serialize_dbcop_history(&to_path)
                    .context("Failed to write DBCop history")?,
                HistoryFormat::Test => history
                    .serialize_test_history(&to_path)
                    .context("Failed to write test history")?,
            }
        }
        Command::Stats {
            path,
            format,
            json_output,
        } => {
            let history = match format {
                HistoryFormat::Plume => History::parse_plume_history(&path)
                    .context("Failed to parse path as Plume history")?,
                HistoryFormat::Cobra => History::parse_cobra_history(&path)
                    .context("Failed to parse path as Cobra history")?,
                #[cfg(feature = "dbcop")]
                HistoryFormat::DbCop => History::parse_dbcop_history(&path)
                    .context("Failed to parse path as Cobra history")?,
                HistoryFormat::Test => History::parse_test_history(&path)
                    .context("Failed to parse path as test history")?,
            };
            if json_output {
                println!("{}", serde_json::to_string_pretty(&history.stats())?);
            } else {
                println!("Stats for {}:", path.display());
                println!("{}", history.stats());
            }
        }
    }
    Ok(())
}

fn pair_to_tid(pair: (usize, isize)) -> Option<TransactionId> {
    if pair.1 < 0 {
        None
    } else {
        Some(TransactionId(pair.0, pair.1 as usize))
    }
}
