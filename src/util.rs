pub trait GetTwoMut<T> {
    fn get_two_mut(&mut self, i: usize, j: usize) -> (&mut T, &mut T);
}

impl<T> GetTwoMut<T> for [T] {
    fn get_two_mut(&mut self, i: usize, j: usize) -> (&mut T, &mut T) {
        if i == j {
            panic!("Attempted to get two mutable references to the same element {i}");
        }

        if i >= self.len() || j >= self.len() {
            panic!("Attempted to get two mutable references to elements outside the slice. indices: ({i}, {j}), len: {}", self.len());
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
