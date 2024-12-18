use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasher, Hash};

use either::Either;

pub trait GetTwoMut<T> {
    fn get_two_mut(&mut self, i: usize, j: usize) -> (&mut T, &mut T);
}

impl<T> GetTwoMut<T> for [T] {
    fn get_two_mut(&mut self, i: usize, j: usize) -> (&mut T, &mut T) {
        if i == j {
            panic!("Attempted to get two mutable references to the same element {i}");
        }

        if i >= self.len() || j >= self.len() {
            panic!("Attempted to get two mutable references to element(s) outside the slice. indices: ({i}, {j}), len: {}", self.len());
        }

        let ptr = self.as_mut_ptr();
        unsafe {
            let a = &mut *ptr.add(i);
            let b = &mut *ptr.add(j);
            (a, b)
        }
    }
}

pub trait Captures<'a> {}

impl<'a, T: ?Sized> Captures<'a> for T {}

pub fn intersect_map<'m, 's, K, V, H>(
    map: &'m HashMap<K, V, H>,
    set: &'s HashSet<K, H>,
) -> impl Iterator<Item = (&'m K, &'m V)> + Captures<'s>
where
    K: Hash + Eq,
    H: BuildHasher,
{
    if map.len() <= set.len() {
        Either::Left(
            map.iter()
                .filter_map(|(k, v)| set.contains(k).then_some((k, v))),
        )
    } else {
        Either::Right(set.iter().filter_map(|k| map.get_key_value(k)))
    }
    .into_iter()
}

struct CompactVec<T> {
    data: VecDeque<T>,
    start: usize,
}

impl<T: Default> CompactVec<T> {
    fn new() -> Self {
        CompactVec {
            data: VecDeque::new(),
            start: 0,
        }
    }

    fn insert(&mut self, index: usize, value: T) {
        if self.data.len() == 0 {
            self.data.push_back(value);
            self.start = index;
        } else if index < self.start {
            for _ in index + 1..self.start {
                self.data.push_front(T::default());
            }
            self.data.push_front(value);
            self.start = index;
        } else if index < self.start + self.data.len() {
            self.data[index - self.start] = value;
        } else {
        }
    }
}
