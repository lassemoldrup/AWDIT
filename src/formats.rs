use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::{fs, io};

#[cfg(feature = "dbcop")]
use dbcop::db::history::Event as DbCopEvent;
#[cfg(feature = "dbcop")]
pub use dbcop::db::history::History as DbCopHistory;
#[cfg(feature = "dbcop")]
use dbcop::db::history::Transaction as DbCopTransaction;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{Event, Key, KeyValuePair, Transaction, Value};

use super::History;

impl History {
    pub fn parse_plume_history(path: impl AsRef<Path>) -> Result<Self, ParseHistoryError> {
        if !path.as_ref().metadata()?.is_dir() {
            return Err(ParseHistoryError::NotADirectory(
                path.as_ref().to_path_buf(),
            ));
        }

        let files: Result<Vec<_>, _> = path.as_ref().read_dir()?.collect();
        let files = files?;
        if files.len() != 1 {
            return Err(ParseHistoryError::NotAPlumeDirectory(
                path.as_ref().to_path_buf(),
            ));
        }

        let contents = fs::read_to_string(files[0].path())?;
        let mut history = History {
            sessions: vec![],
            aborted_writes: FxHashSet::default(),
        };
        let mut session_map = FxHashMap::default();
        let mut transaction_map = FxHashMap::default();
        let mut keys = FxHashSet::default();

        for e in contents.lines() {
            let (op, e) = e
                .split_at_checked(1)
                .ok_or(ParseHistoryError::InvalidPlumeFormat)?;
            let e = e
                .strip_prefix('(')
                .and_then(|e| e.strip_suffix(')'))
                .ok_or(ParseHistoryError::InvalidPlumeFormat)?;
            let mut parts = e.split(',');
            let key = parts
                .next()
                .and_then(|k| k.parse::<usize>().ok())
                .ok_or(ParseHistoryError::InvalidPlumeFormat)?;
            let val = parts
                .next()
                .and_then(|k| k.parse::<usize>().ok())
                .ok_or(ParseHistoryError::InvalidPlumeFormat)?;
            let sess = parts
                .next()
                .and_then(|k| k.parse::<u64>().ok())
                .ok_or(ParseHistoryError::InvalidPlumeFormat)?;
            let txn = parts
                .next()
                .and_then(|k| k.parse::<i64>().ok())
                .ok_or(ParseHistoryError::InvalidPlumeFormat)?;

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

    pub fn serialize_plume_history(&self, path: impl AsRef<Path>) -> Result<(), io::Error> {
        let file_path = path.as_ref().join("history.txt");
        fs::create_dir_all(path)?;
        let mut writer = File::create(file_path)?;
        write!(&mut writer, "{}", PlumeHistoryDisplay { history: self })
    }

    pub fn parse_cobra_history(path: impl AsRef<Path>) -> Result<Self, ParseHistoryError> {
        if !path.as_ref().metadata()?.is_dir() {
            return Err(ParseHistoryError::NotADirectory(
                path.as_ref().to_path_buf(),
            ));
        }

        let mut history = History {
            sessions: vec![],
            aborted_writes: FxHashSet::default(),
        };
        let mut keys = FxHashSet::default();

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

                        session.push(cur_txn);
                    }
                    b'A' => {
                        // There shouldn't be any aborted transactions in the log
                        return Err(ParseHistoryError::InvalidCobraFormat);
                    }
                    b'W' => {
                        let Some(cur_txn) = &mut cur_txn else {
                            return Err(ParseHistoryError::InvalidCobraFormat);
                        };

                        let mut buf = [0; 8];
                        reader.read_exact(&mut buf)?;
                        let wid = i64::from_be_bytes(buf);
                        reader.read_exact(&mut buf)?;
                        let key_hash = i64::from_be_bytes(buf);
                        reader.read_exact(&mut buf)?;
                        // TODO: use this for anything?
                        let _val_hash = i64::from_be_bytes(buf);

                        cur_txn.push(Event::Write(KeyValuePair {
                            key: Key(key_hash as usize),
                            value: Value(wid as usize),
                        }));
                    }
                    b'R' => {
                        let Some(cur_txn) = &mut cur_txn else {
                            return Err(ParseHistoryError::InvalidCobraFormat);
                        };

                        let mut buf = [0; 8];
                        reader.read_exact(&mut buf)?;
                        // TODO: use this for anything?
                        let _prev_txnid = i64::from_be_bytes(buf);
                        reader.read_exact(&mut buf)?;
                        let mut wid = i64::from_be_bytes(buf);
                        reader.read_exact(&mut buf)?;
                        let key_hash = i64::from_be_bytes(buf);
                        reader.read_exact(&mut buf)?;
                        // TODO: use this for anything?
                        let _val_hash = i64::from_be_bytes(buf);

                        let key = Key(key_hash as usize);
                        keys.insert(key);

                        if key.0 == 13296660918851520211 {
                            dbg!(_prev_txnid, wid, key_hash, _val_hash);
                        }

                        // Null reads are treated as init reads
                        if wid == 0xdeadbeef || wid == 0xbebeebee {
                            wid = 0;
                        }

                        cur_txn.push(Event::Read(KeyValuePair {
                            key: Key(key_hash as usize),
                            value: Value(wid as usize),
                        }));
                    }
                    _ => return Err(ParseHistoryError::InvalidCobraFormat),
                }
            }

            history.sessions.push(session);
        }

        // TODO: maybe handle this implicitly?
        let mut init_transaction = Transaction::new();
        for &key in &keys {
            init_transaction.push(Event::Write(KeyValuePair {
                key,
                value: Value(0),
            }));
        }
        history.sessions.push(vec![init_transaction]);

        if history.sessions.is_empty() {
            return Err(ParseHistoryError::InvalidCobraFormat);
        }

        Ok(history)
    }

    #[cfg(feature = "dbcop")]
    pub fn parse_dbcop_history(path: impl AsRef<Path>) -> Result<Self, ParseHistoryError> {
        if !path.as_ref().metadata()?.is_dir() {
            return Err(ParseHistoryError::NotADirectory(
                path.as_ref().to_path_buf(),
            ));
        }

        let file_path = path.as_ref().join("history.bincode");
        let reader = BufReader::new(File::open(file_path)?);
        let history: DbCopHistory =
            bincode::deserialize_from(reader).map_err(|_| ParseHistoryError::InvalidDbCopFormat)?;
        Ok(Self::from_dbcop_history(&history))
    }

    #[cfg(feature = "dbcop")]
    pub fn from_dbcop_history(history: &DbCopHistory) -> Self {
        let dbcop_sessions = history.get_data();
        let mut sessions = Vec::with_capacity(dbcop_sessions.len());
        let mut aborted_writes = FxHashSet::default();
        let mut keys = FxHashSet::default();
        for dbcop_session in dbcop_sessions {
            let mut session = Vec::with_capacity(dbcop_session.len());
            for dbcop_txn in dbcop_session {
                let mut txn = Transaction::new();
                for event in &dbcop_txn.events {
                    let kv = KeyValuePair {
                        key: Key(event.variable),
                        value: Value(event.value),
                    };
                    keys.insert(kv.key);
                    if event.write {
                        if !event.success {
                            aborted_writes.insert(kv);
                            continue;
                        }
                        txn.push(Event::Write(kv));
                    } else {
                        if !event.success {
                            continue;
                        }
                        txn.push(Event::Read(kv));
                    }
                }
                session.push(txn);
            }
            sessions.push(session);
        }

        // TODO: maybe handle this implicitly?
        let mut init_transaction = Transaction::new();
        for key in keys {
            init_transaction.push(Event::Write(KeyValuePair {
                key,
                value: Value(0),
            }));
        }
        sessions.push(vec![init_transaction]);

        History {
            sessions,
            aborted_writes,
        }
    }

    #[cfg(feature = "dbcop")]
    pub fn serialize_dbcop_history(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(), SerializeDbCopHistoryError> {
        let dbcop_history = self.to_dbcop_history();
        let file_path = path.as_ref().join("history.bincode");
        fs::create_dir_all(path)?;
        let writer = File::create(file_path)?;
        bincode::serialize_into(writer, &dbcop_history).map_err(Into::into)
    }

    #[cfg(feature = "dbcop")]
    pub fn to_dbcop_history(&self) -> DbCopHistory {
        let mut dbcop_sessions = Vec::with_capacity(self.sessions.len() + 1);
        for session in &self.sessions {
            let mut dbcop_session = Vec::with_capacity(session.len());
            for txn in session {
                let mut dbcop_txn = DbCopTransaction {
                    events: Vec::with_capacity(txn.events.len()),
                    success: true,
                };

                for event in &txn.events {
                    let (variable, value) = match event {
                        Event::Read(kv) => (kv.key.0, kv.value.0),
                        Event::Write(kv) => (kv.key.0, kv.value.0),
                    };
                    dbcop_txn.events.push(dbcop::db::history::Event {
                        variable,
                        value,
                        write: matches!(event, Event::Write(_)),
                        success: true,
                    });
                }
                dbcop_session.push(dbcop_txn);
            }
            dbcop_sessions.push(dbcop_session);
        }

        if !self.aborted_writes.is_empty() {
            let mut dbcop_txn = DbCopTransaction {
                events: Vec::with_capacity(self.aborted_writes.len()),
                success: false,
            };
            for &kv in &self.aborted_writes {
                dbcop_txn.events.push(DbCopEvent {
                    variable: kv.key.0,
                    value: kv.value.0,
                    write: true,
                    success: false,
                });
            }
            dbcop_sessions.push(vec![dbcop_txn]);
        }

        DbCopHistory::new(
            self.stats().to_hist_params(),
            "converted".into(),
            chrono::Local::now(),
            chrono::Local::now(),
            dbcop_sessions,
        )
    }

    pub fn serialize_test_history(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(), SerializeTestHistoryError> {
        let mut writer = File::create(path)?;
        write!(&mut writer, "{}", self)?;
        Ok(())
    }

    pub fn parse_test_history(path: impl AsRef<Path>) -> io::Result<Self> {
        let contents = fs::read_to_string(path)?;
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
        Ok(Self {
            sessions,
            aborted_writes: FxHashSet::default(),
        })
    }
}

struct PlumeHistoryDisplay<'h> {
    history: &'h History,
}

impl Display for PlumeHistoryDisplay<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for aborted in &self.history.aborted_writes {
            writeln!(f, "w({},{},0,-1)", aborted.key, aborted.value)?;
        }
        let mut t_idx = 0;
        for (s_idx, session) in self.history.sessions.iter().enumerate() {
            for txn in session {
                for event in &txn.events {
                    match event {
                        Event::Read(kv) => {
                            writeln!(f, "r({},{},{},{})", kv.key, kv.value, s_idx, t_idx)?
                        }
                        Event::Write(kv) => {
                            writeln!(f, "w({},{},{},{})", kv.key, kv.value, s_idx, t_idx)?
                        }
                    }
                }
                t_idx += 1;
            }
        }

        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ParseHistoryError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("{0} is not a directory")]
    NotADirectory(PathBuf),
    #[error("{0} is not a single-file directory with a .txt file")]
    NotAPlumeDirectory(PathBuf),
    #[error("File did not match Plume format")]
    InvalidPlumeFormat,
    #[error("File did not match Cobra format")]
    InvalidCobraFormat,
    #[cfg(feature = "dbcop")]
    #[error("File did not match DBCop format")]
    InvalidDbCopFormat,
}

#[derive(thiserror::Error, Debug)]
pub enum SerializeTestHistoryError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("Path is not a file")]
    PathError,
}

#[cfg(feature = "dbcop")]
#[derive(thiserror::Error, Debug)]
pub enum SerializeDbCopHistoryError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("{0}")]
    Bincode(#[from] bincode::Error),
}
