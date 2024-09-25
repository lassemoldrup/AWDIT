use std::collections::BTreeMap;

use clap::{Parser, ValueEnum};
use consistency::util::GetTwoMut;
use consistency::vector_clock::VectorClock;
use consistency::{Event, History, Key, Transaction, TransactionId, Value};
use rand::prelude::*;
use rand_distr::{Bernoulli, Pareto, Uniform};
use rustc_hash::FxHashMap;

#[derive(Parser)]
struct App {
    #[clap(short, long, default_value_t = 8)]
    locations: usize,
    #[clap(short, long, default_value_t = 20)]
    events: usize,
    #[clap(short, long, default_value_t = IsolationLevel::Causal)]
    isolation: IsolationLevel,
    #[clap(short, long, default_value_t = 0.8)]
    read_ratio: f64,
    #[clap(short, long, default_value_t = 5.)]
    mean_transaction_size: f64,
    #[clap(short, long, default_value_t = 1.16)]
    alpha: f64,
}

impl App {
    fn _find_inconsistent(&self) {
        for _ in 0..100000 {
            let history = generate_causal_history(self);
            if !history.check_causal() {
                println!("{history}");
                return;
            }
        }
        println!("Failed to find inconsistent")
    }
}

#[derive(Clone, PartialEq, Eq, ValueEnum, strum::EnumString, strum::Display)]
enum IsolationLevel {
    #[strum(serialize = "read-committed")]
    ReadCommitted,
    #[strum(serialize = "read-atomic")]
    ReadAtomic,
    #[strum(serialize = "causal")]
    Causal,
}

fn main() {
    let app = App::parse();
    let history = generate_causal_history(&app);
    println!("{history}");
}

fn generate_causal_history(app: &App) -> History {
    if app.isolation != IsolationLevel::Causal {
        panic!("Only causal isolation is supported");
    }
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
        .extend((0..app.locations).map(|k| Event::Write(Key(k), Value(0))));

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
                Event::Write(k, _) if *k == key => should_commit = true,
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
            Event::Read(key, Value(0))
        } else {
            let value = *value_map.entry(key).and_modify(|v| *v += 1).or_insert(1);
            Event::Write(key, Value(value))
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

    // Update read events
    let mut hb = sessions
        .iter()
        .map(|s| vec![VectorClock::new_min(sessions.len()); s.len()])
        .collect::<Vec<_>>();
    let mut writes_per_loc_per_session: Vec<FxHashMap<Key, Vec<usize>>> =
        vec![FxHashMap::default(); sessions.len()];
    // Values written by transactions in commit order
    let mut writes_per_loc: Vec<Vec<(TransactionId, Value)>> = vec![vec![]; app.locations];
    let mut idx_in_writes_per_loc = vec![FxHashMap::default(); app.locations];
    for tid @ &TransactionId(s_idx, t_idx) in commit_order.iter() {
        if t_idx > 0 {
            let (pred, succ) = hb[s_idx].get_two_mut(t_idx - 1, t_idx);
            succ.join(&pred);
            succ.join1(s_idx, t_idx as isize - 1);
        }
        let mut read_map = FxHashMap::default();
        let mut reads: BTreeMap<usize, Vec<Key>> = BTreeMap::new();
        for event in sessions[s_idx][t_idx].events.iter_mut() {
            match event {
                Event::Read(k, v) => {
                    // We implement repeatable reads explicitly as an optimization
                    if let Some(&(_, write_value)) = read_map.get(k) {
                        *v = write_value;
                        continue;
                    }
                    // Find co-first write that the read is allowed to read from
                    let latest = hb[s_idx][t_idx]
                        .iter()
                        .enumerate()
                        .filter_map(pair_to_tid)
                        .filter_map(|tid| {
                            // Last write to k in the session
                            writes_per_loc_per_session[tid.0].get(&*k).and_then(|ws| {
                                ws.binary_search(&tid.1)
                                    .or_else(|i| i.checked_sub(1).ok_or(()))
                                    .ok()
                                    .map(|i| TransactionId(tid.0, ws[i]))
                            })
                        })
                        .max_by(|t1, t2| rev_commit_order[t1].cmp(&rev_commit_order[t2]))
                        .unwrap_or(TransactionId(num_sessions, 0));
                    let idx = idx_in_writes_per_loc[k.0][&latest];

                    // Use rejection sampling to determine a choice out of the remaining possibilities
                    let num_choices = writes_per_loc[k.0].len() - idx;
                    let mut choices = (0..num_choices).collect::<Vec<_>>();
                    choices.shuffle(&mut rng);
                    let mut consistent_choice = None;
                    'choice_loop: for choice in choices {
                        let (write_tid, write_value) = writes_per_loc[k.0][idx + choice];
                        let write_co_idx = rev_commit_order[&write_tid];

                        // Optimization: we know it's safe to read from writes co-before any of our other writers
                        // Optimization: if we are already reading from a transaction, we can do so again
                        let co_min_read = reads
                            .first_key_value()
                            .map(|(i, _)| *i)
                            .unwrap_or(usize::MAX);
                        if write_co_idx <= co_min_read || reads.contains_key(&write_co_idx) {
                            consistent_choice = Some((write_tid, write_value));
                            break;
                        }

                        // Other writers before the considered write could cause problems
                        for (&other_write_co_idx, keys) in reads.range(..write_co_idx) {
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
                                        writes_per_loc_per_session[tid.0].get(&k2).and_then(|ws| {
                                            ws.binary_search(&tid.1)
                                                .or_else(|i| i.checked_sub(1).ok_or(()))
                                                .ok()
                                                .map(|i| TransactionId(tid.0, ws[i]))
                                    }))
                                })
                                .any(|tid| rev_commit_order[&tid] > other_write_co_idx);
                            if inconsistent {
                                continue 'choice_loop;
                            }
                        }

                        consistent_choice = Some((write_tid, write_value));
                    }

                    let (write_tid, write_value) =
                        consistent_choice.expect("Should have at least one valid writer");
                    *v = write_value;
                    reads
                        .entry(rev_commit_order[&write_tid])
                        .or_default()
                        .push(*k);
                    read_map.insert(*k, (write_tid, write_value));
                    hb[s_idx][t_idx].join1(write_tid.0, write_tid.1 as isize);
                    if write_tid.0 != s_idx {
                        let (pred_sess, succ_sess) = hb.get_two_mut(write_tid.0, s_idx);
                        succ_sess[t_idx].join(&pred_sess[write_tid.1]);
                    }
                }
                Event::Write(k, v) => {
                    writes_per_loc_per_session[s_idx]
                        .entry(*k)
                        .or_default()
                        .push(t_idx);
                    idx_in_writes_per_loc[k.0].insert(*tid, writes_per_loc[k.0].len());
                    writes_per_loc[k.0].push((*tid, *v));
                }
            }
        }
    }

    sessions.retain(|s| !s.is_empty());
    sessions.shuffle(&mut rng);
    History { sessions }
}

fn pair_to_tid(pair: (usize, isize)) -> Option<TransactionId> {
    if pair.1 < 0 {
        None
    } else {
        Some(TransactionId(pair.0, pair.1 as usize))
    }
}
