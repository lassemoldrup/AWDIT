use std::iter;

use crate::fenwick::MinFenwickTree;
use crate::util::Captures;
use crate::TransactionId;

/// A data structure for storing partial orders over transactions in `k` sessions.
/// Supports `O(k^2 log n)` edge insertion and `O(k^2 log n)` reachability queries.
#[derive(Debug)]
pub struct PartialOrder {
    /// `edges[i][j]` holds all edges from session `i` to session `j`.
    edges: Vec<Vec<MinFenwickTree<usize>>>,
}

impl PartialOrder {
    pub fn new(sessions: Vec<usize>) -> Self {
        let mut edges = vec![vec![]; sessions.len()];
        for (i1, &sess_len) in sessions.iter().enumerate() {
            for i2 in 0..sessions.len() {
                if i1 == i2 {
                    // Edges to the same session point to the next transaction, or usize::MAX for
                    // the last event. This makes sure that transaction can reach other transactions
                    // in the same session.
                    edges[i1].push(MinFenwickTree::build((1..sess_len).chain([usize::MAX])));
                } else {
                    edges[i1].push(MinFenwickTree::build(
                        iter::repeat(usize::MAX).take(sess_len),
                    ));
                }
            }
        }

        Self { edges }
    }

    pub fn insert(
        &mut self,
        t1: TransactionId,
        t2: TransactionId,
    ) -> Result<(), PartialOrderCycleError> {
        // Check for cycles.
        if self.successor(t2, t1.0) <= t1.1 {
            return Err(PartialOrderCycleError);
        }

        let num_sessions = self.edges.len();
        for i1 in 0..num_sessions {
            for i2 in 0..num_sessions {
                let Some(j1) = self.predecessor(t1, i1) else {
                    continue;
                };
                let j2 = self.successor(t2, i2);
                self.edges[i1][i2].update(j1, j2);
            }
        }

        Ok(())
    }

    pub fn query(&self, t1: TransactionId, t2: TransactionId) -> bool {
        self.successor(t1, t2.0) <= t2.1
    }

    pub fn first_reachable(&self, t1: TransactionId, s_idx: usize) -> Option<TransactionId> {
        let j = self.successor(t1, s_idx);
        if j < self.edges[s_idx].len() {
            Some(TransactionId(s_idx, j))
        } else {
            None
        }
    }

    pub fn successors<'p>(
        &'p self,
        t: TransactionId,
    ) -> impl Iterator<Item = TransactionId> + Captures<'p> {
        (0..self.edges.len()).flat_map(move |i2| {
            let num_txs = self.edges[i2].len();
            let j2 = self.edges[t.0][i2].query(t.1).min(num_txs);
            (j2..num_txs).map(move |j| TransactionId(i2, j))
        })
    }

    fn successor(&self, TransactionId(i, j): TransactionId, s_idx: usize) -> usize {
        // TODO: Should we have self loops by default?
        if i == s_idx {
            j
        } else {
            self.edges[i][s_idx].query(j)
        }
    }

    fn predecessor(&self, TransactionId(i, j): TransactionId, s_idx: usize) -> Option<usize> {
        if i == s_idx {
            Some(j)
        } else {
            self.edges[s_idx][i].arg_leq(j)
        }
    }
}

#[derive(Debug)]
pub struct PartialOrderCycleError;

#[cfg(test)]
mod test {
    use itertools::iproduct;

    use super::*;
    use crate::TransactionId as Tid;

    #[test]
    fn partial_order_two_session_test() {
        // Two sessions with indices 0, 1, 2 || 3, 4, 5
        let mut partial_order = PartialOrder::new(vec![3, 3]);
        assert!(partial_order.query(Tid(0, 0), Tid(0, 2)));
        assert!(!partial_order.query(Tid(0, 0), Tid(1, 0)));

        // Insert 0 -> 3
        assert!(partial_order.insert(Tid(0, 0), Tid(1, 0)).is_ok());
        assert!(partial_order.query(Tid(0, 0), Tid(1, 0)));
        assert!(partial_order.query(Tid(0, 0), Tid(1, 2)));
        assert!(!partial_order.query(Tid(0, 1), Tid(1, 1)));

        // Insert 1 -> 4
        assert!(partial_order.insert(Tid(0, 1), Tid(1, 1)).is_ok());
        assert!(partial_order.query(Tid(0, 1), Tid(1, 1)));
        assert!(partial_order.query(Tid(0, 1), Tid(1, 2)));
        assert!(!partial_order.query(Tid(0, 2), Tid(1, 2)));
        assert!(!partial_order.query(Tid(1, 2), Tid(0, 2)));

        // Insert 5 -> 2
        assert!(partial_order.insert(Tid(1, 2), Tid(0, 2)).is_ok());
        assert!(partial_order.query(Tid(1, 2), Tid(0, 2)));
        assert!(partial_order.query(Tid(1, 1), Tid(0, 2)));
        assert!(!partial_order.query(Tid(1, 0), Tid(0, 1)));
        assert!(!partial_order.query(Tid(0, 2), Tid(1, 2)));

        // Sanity check
        assert!(partial_order.query(Tid(0, 0), Tid(1, 2)));

        // Cycle checks
        assert!(partial_order.insert(Tid(0, 0), Tid(0, 0)).is_err());
        assert!(partial_order.insert(Tid(1, 2), Tid(1, 0)).is_err());
        assert!(partial_order.insert(Tid(0, 2), Tid(1, 1)).is_err());
    }

    #[test]
    fn partial_order_three_session_test() {
        // Three sessions with indices 0, 1, 2, || 3, 4, 5, || 6, 7, 8
        let mut partial_order = PartialOrder::new(vec![3, 3, 3]);
        assert!(partial_order.query(Tid(0, 0), Tid(0, 2)));
        assert!(!partial_order.query(Tid(0, 0), Tid(2, 0)));

        // 0----->3 ||  6
        //    ||--------^
        // 1----^ 4<----7
        //    ||    ||
        // 2  ||  5---->8
        assert!(partial_order.insert(Tid(0, 0), Tid(1, 0)).is_ok());
        assert!(partial_order.insert(Tid(0, 1), Tid(2, 0)).is_ok());
        assert!(partial_order.insert(Tid(2, 1), Tid(1, 1)).is_ok());
        assert!(partial_order.insert(Tid(1, 2), Tid(2, 2)).is_ok());

        // 0 can reach everything
        for i in 0..3 {
            for j in 0..3 {
                assert!(partial_order.query(Tid(0, 0), Tid(i, j)));
            }
        }

        assert!(partial_order.query(Tid(0, 1), Tid(2, 1)));
        assert!(partial_order.query(Tid(0, 1), Tid(1, 2)));
        assert!(!partial_order.query(Tid(0, 1), Tid(1, 0)));

        assert!(partial_order.query(Tid(2, 0), Tid(1, 1)));
        assert!(!partial_order.query(Tid(2, 0), Tid(0, 2)));

        // Cycle checks
        assert!(partial_order.insert(Tid(2, 2), Tid(1, 0)).is_err());
        assert!(partial_order.insert(Tid(2, 2), Tid(0, 1)).is_err());
        assert!(partial_order.insert(Tid(1, 1), Tid(0, 0)).is_err());
        assert!(partial_order.insert(Tid(1, 1), Tid(2, 0)).is_err());
    }

    #[test]
    fn partial_order_iter_test() {
        // Three sessions with indices 0, 1, 2, || 3, 4, 5, || 6, 7, 8
        let mut partial_order = PartialOrder::new(vec![3, 3, 3]);

        // 0----->3 ||  6
        //    ||--------^
        // 1----^ 4<----7
        //    ||    ||
        // 2  ||  5---->8
        partial_order.insert(Tid(0, 0), Tid(1, 0)).unwrap();
        partial_order.insert(Tid(0, 1), Tid(2, 0)).unwrap();
        partial_order.insert(Tid(2, 1), Tid(1, 1)).unwrap();
        partial_order.insert(Tid(1, 2), Tid(2, 2)).unwrap();

        itertools::assert_equal(
            partial_order.successors(Tid(0, 0)),
            iproduct!(0..3, 0..3).map(|(i, j)| Tid(i, j)).skip(1),
        );
        itertools::assert_equal(
            partial_order.successors(Tid(0, 1)),
            vec![
                Tid(0, 2),
                Tid(1, 1),
                Tid(1, 2),
                Tid(2, 0),
                Tid(2, 1),
                Tid(2, 2),
            ],
        );
        itertools::assert_equal(
            partial_order.successors(Tid(2, 0)),
            vec![Tid(1, 1), Tid(1, 2), Tid(2, 1), Tid(2, 2)],
        );
    }
}
