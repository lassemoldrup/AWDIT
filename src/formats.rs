use std::path::Path;
use std::sync::LazyLock;
use std::{fs, io};

use regex::Regex;
use rustc_hash::FxHashMap;

use crate::{Event, Key, KeyValuePair, Transaction, Value};

use super::History;

static PLUME_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([rw])\((\d++),(\d++),(\d++),(-?\d++)\)").unwrap());

impl History {
    pub fn parse_plume_history(path: impl AsRef<Path>) -> Result<Self, ParseHistoryError> {
        let contents = fs::read_to_string(path)?;
        let mut history = History { sessions: vec![] };
        let mut session_map = FxHashMap::default();
        let mut transaction_map = FxHashMap::default();

        for e in contents.lines() {
            let (_, [op, key, val, sess, txn]) = PLUME_REGEX
                .captures(e)
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

            let kv = KeyValuePair {
                key: Key(key),
                value: Value(val),
            };
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

        Ok(history)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ParseHistoryError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("File did not match Plume format")]
    InvalidPlumeFormat,
    #[error("Aborted transactions are currently not supported")]
    AbortedNotSupported,
}
