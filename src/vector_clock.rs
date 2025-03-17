use std::cmp;
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorClock(Box<[i32]>);

impl VectorClock {
    pub fn new_min(size: usize) -> Self {
        VectorClock(vec![-1; size].into_boxed_slice())
    }

    pub fn new_max(size: usize) -> Self {
        VectorClock(vec![i32::MAX; size].into_boxed_slice())
    }

    pub fn join(&mut self, other: &VectorClock) {
        assert_eq!(self.0.len(), other.0.len());
        for (i, j) in self.0.iter_mut().zip(&other.0) {
            *i = cmp::max(*i, *j);
        }
    }

    pub fn join1(&mut self, index: usize, value: i32) {
        let entry = &mut self[index];
        *entry = cmp::max(*entry, value as i32);
    }

    pub fn meet(&mut self, other: &VectorClock) {
        assert_eq!(self.0.len(), other.0.len());
        for (i, j) in self.0.iter_mut().zip(&other.0) {
            *i = cmp::min(*i, *j);
        }
    }

    pub fn meet1(&mut self, index: usize, value: i32) {
        let entry = &mut self[index];
        *entry = cmp::min(*entry, value as i32);
    }

    pub fn iter(&self) -> impl Iterator<Item = i32> {
        self.0.iter().copied()
    }
}

impl Index<usize> for VectorClock {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for VectorClock {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
