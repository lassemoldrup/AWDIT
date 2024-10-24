use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::{fs, io};

use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{Event, Key, KeyValuePair, Transaction, TransactionId, Value};

use super::History;

thread_local! {
    static PLUME_REGEX: Regex = Regex::new(r"([rw])\((\d++),(\d++),(\d++),(-?\d++)\)").unwrap();
}

impl History {
    pub fn parse_plume_history(path: impl AsRef<Path>) -> Result<Self, ParseHistoryError> {
        let contents = fs::read_to_string(path)?;
        let mut history = History {
            sessions: vec![],
            aborted_writes: FxHashSet::default(),
        };
        let mut session_map = FxHashMap::default();
        let mut transaction_map = FxHashMap::default();
        let mut keys = FxHashSet::default();

        for e in contents.lines() {
            let (_, [op, key, val, sess, txn]) = PLUME_REGEX
                .with(|r| r.captures(e))
                .ok_or(ParseHistoryError::InvalidPlumeFormat)?
                .extract();
            let key = key
                .parse::<usize>()
                .map_err(|_| ParseHistoryError::InvalidPlumeFormat)?;
            let val = val
                .parse::<usize>()
                .map_err(|_| ParseHistoryError::InvalidPlumeFormat)?;
            let sess = sess
                .parse::<u64>()
                .map_err(|_| ParseHistoryError::InvalidPlumeFormat)?;
            let txn = txn
                .parse::<i64>()
                .map_err(|_| ParseHistoryError::InvalidPlumeFormat)?;

            keys.insert(Key(key));
            let kv = KeyValuePair {
                key: Key(key),
                value: Value(val),
            };

            if txn == -1 {
                if op == "w" {
                    history.aborted_writes.insert(kv);
                }
                continue;
            }

            let event = match op {
                "r" => Event::Read(kv),
                "w" => Event::Write(kv),
                _ => return Err(ParseHistoryError::InvalidPlumeFormat),
            };

            let s_idx = *session_map.entry(sess).or_insert_with(|| {
                history.sessions.push(Vec::new());
                history.sessions.len() - 1
            });
            let t_idx = *transaction_map.entry(txn).or_insert_with(|| {
                history.sessions[s_idx].push(Transaction::new());
                history.sessions[s_idx].len() - 1
            });
            history.sessions[s_idx][t_idx].events.push(event);
        }

        // TODO: maybe handle this implicitly?
        let mut init_transaction = Transaction::new();
        for key in keys {
            init_transaction.push(Event::Write(KeyValuePair {
                key,
                value: Value(0),
            }));
        }
        history.sessions.push(vec![init_transaction]);

        Ok(history)
    }

    pub fn parse_cobra_history(path: impl AsRef<Path>) -> Result<Self, ParseHistoryError> {
        let mut history = History {
            sessions: vec![],
            aborted_writes: FxHashSet::default(),
        };
        let mut txn_id_map = FxHashMap::default();
        for log_file in fs::read_dir(path)? {
            let file_path = log_file?.path();
            if file_path.extension().and_then(|e| e.to_str()) != Some("log") {
                continue;
            }

            let mut reader = BufReader::new(File::open(file_path)?);
            let mut session = Vec::new();
            let mut cur_txn = None;
            let mut cur_txn_id = None;

            while let Some(op_type) = (&mut reader).bytes().next().transpose()? {
                match op_type {
                    b'S' => {
                        if cur_txn.is_some() {
                            return Err(ParseHistoryError::InvalidCobraFormat);
                        }

                        let mut txn_id = [0; 8];
                        reader.read_exact(&mut txn_id)?;
                        let txn_id = i64::from_be_bytes(txn_id);

                        cur_txn = Some(Transaction::new());
                        cur_txn_id = Some(txn_id);
                    }
                    b'C' => {
                        let (Some(cur_txn), Some(cur_txn_id)) = (cur_txn.take(), cur_txn_id.take())
                        else {
                            return Err(ParseHistoryError::InvalidCobraFormat);
                        };

                        let mut c_txn_id = [0; 8];
                        reader.read_exact(&mut c_txn_id)?;
                        let c_txn_id = i64::from_be_bytes(c_txn_id);
                        if c_txn_id != cur_txn_id {
                            return Err(ParseHistoryError::InvalidCobraFormat);
                        }

                        txn_id_map.insert(
                            cur_txn_id,
                            TransactionId(history.sessions.len(), session.len()),
                        );
                        session.push(cur_txn);
                    }
                    b'A' => {
                        // There shouldn't be any aborted transactions in the log
                        return Err(ParseHistoryError::InvalidCobraFormat);
                    }
                    b'W' => {}
                    b'R' => {}
                    _ => return Err(ParseHistoryError::InvalidCobraFormat),
                }
            }

            history.sessions.push(session);
        }

        if history.sessions.is_empty() {
            return Err(ParseHistoryError::InvalidCobraFormat);
        }

        Ok(history)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ParseHistoryError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("File did not match Plume format")]
    InvalidPlumeFormat,
    #[error("File did not match Cobra format")]
    InvalidCobraFormat,
}
