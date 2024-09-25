use std::cmp;
use std::ops::Index;

use crate::util::Captures;

#[derive(Clone, Debug)]
pub struct VectorClock(Box<[isize]>);

impl VectorClock {
    pub fn new_min(size: usize) -> Self {
        VectorClock(vec![-1; size].into_boxed_slice())
    }

    pub fn new_max(size: usize) -> Self {
        VectorClock(vec![isize::MAX; size].into_boxed_slice())
    }

    // TODO: SIMD
    pub fn join(&mut self, other: &VectorClock) {
        assert_eq!(self.0.len(), other.0.len());
        for (i, j) in self.0.iter_mut().zip(&other.0) {
            *i = cmp::max(*i, *j);
        }
    }

    pub fn join1(&mut self, index: usize, value: isize) {
        self.0[index] = cmp::max(self.0[index], value);
    }

    // TODO: SIMD
    pub fn meet(&mut self, other: &VectorClock) {
        assert_eq!(self.0.len(), other.0.len());
        for (i, j) in self.0.iter_mut().zip(&other.0) {
            *i = cmp::min(*i, *j);
        }
    }

    pub fn meet1(&mut self, index: usize, value: isize) {
        self.0[index] = cmp::min(self.0[index], value);
    }

    pub fn iter(&self) -> impl Iterator<Item = isize> + Captures<'_> {
        self.0.iter().copied()
    }
}

impl Index<usize> for VectorClock {
    type Output = isize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
