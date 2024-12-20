use std::cmp;
use std::ops::{Index, IndexMut};

use wide::i32x4;

use crate::util::Captures;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorClock(Box<[i32x4]>);

impl VectorClock {
    pub fn new_min(size: usize) -> Self {
        VectorClock(vec![i32x4::splat(-1); div_up(size, 4)].into_boxed_slice())
    }

    pub fn new_max(size: usize) -> Self {
        VectorClock(vec![i32x4::MAX; div_up(size, 4)].into_boxed_slice())
    }

    // TODO: SIMD
    pub fn join(&mut self, other: &VectorClock) {
        assert_eq!(self.0.len(), other.0.len());
        for (i, j) in self.0.iter_mut().zip(&other.0) {
            *i = i.max(*j);
        }
    }

    pub fn join1(&mut self, index: usize, value: isize) {
        let entry = &mut self[index];
        *entry = cmp::max(*entry, value as i32);
    }

    // TODO: SIMD
    pub fn meet(&mut self, other: &VectorClock) {
        assert_eq!(self.0.len(), other.0.len());
        for (i, j) in self.0.iter_mut().zip(&other.0) {
            *i = i.min(*j)
        }
    }

    pub fn meet1(&mut self, index: usize, value: isize) {
        let entry = &mut self[index];
        *entry = cmp::min(*entry, value as i32);
    }

    pub fn iter(&self) -> impl Iterator<Item = isize> + Captures<'_> {
        self.0
            .iter()
            .flat_map(|x| x.as_array_ref().iter().copied().map(|v| v as isize))
    }
}

impl Index<usize> for VectorClock {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index / 4].as_array_ref()[index % 4]
    }
}

impl IndexMut<usize> for VectorClock {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index / 4].as_array_mut()[index % 4]
    }
}

fn div_up(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}
