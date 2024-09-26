use std::collections::{HashMap, HashSet};
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
) -> impl Iterator<Item = &'m V> + Captures<'s>
where
    K: Hash + Eq,
    H: BuildHasher,
{
    if map.len() <= set.len() {
        Either::Left(map.iter().filter_map(|(k, v)| set.contains(k).then_some(v)))
    } else {
        Either::Right(set.iter().filter_map(|k| map.get(k)))
    }
    .into_iter()
}
