use proconio::{derive_readable, input, source::line::LineSource};

const N: usize = 200;
const MIN_Z: i32 = 10;
const MAX_Z: i32 = 5000;

#[derive_readable]
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}
impl std::ops::Add for Position {
    type Output = Self;

    fn add(self, other: Position) -> Self {
        Position {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
impl std::ops::Sub for Position {
    type Output = Self;

    fn sub(self, other: Position) -> Self {
        Position {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}
impl Position {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
    fn manhattan_distance(&self, other: &Position) -> i32 {
        let diff = *self - *other;
        diff.x.abs() + diff.y.abs()
    }
}

#[derive(Debug, Clone)]
struct Pixel {
    is_broken: bool,
    power_consumed: i32,
}
impl Pixel {
    fn new() -> Self {
        Self {
            is_broken: false,
            power_consumed: 0,
        }
    }
}

#[derive(Debug)]
struct Map<'a> {
    querier: querier::Querier<'a>,
    state: Vec<Vec<Pixel>>,
}
impl<'a> Map<'a> {
    fn new(querier: querier::Querier<'a>) -> Self {
        Self {
            querier,
            state: vec![vec![Pixel::new(); N]; N],
        }
    }
    fn ref_pixel(&self, position: &Position) -> &Pixel {
        &self.state[position.x as usize][position.y as usize]
    }
    fn ref_mut_pixel(&mut self, position: &Position) -> &mut Pixel {
        &mut self.state[position.x as usize][position.y as usize]
    }
    fn is_broken(&self, position: &Position) -> bool {
        self.ref_pixel(position).is_broken
    }
    fn dig(&mut self, position: &Position, power: i32) -> bool {
        if self.is_broken(position) {
            return true;
        }
        let broke_after_dig = self.querier.query(position, power);
        let mut target_pixel = self.ref_mut_pixel(position);
        if broke_after_dig {
            target_pixel.is_broken = true;
        }
        target_pixel.power_consumed += power;
        broke_after_dig
    }
    fn dig_until_break(&mut self, position: &Position, power: i32) {
        loop {
            if self.dig(position, power) {
                return;
            }
        }
    }
}

fn main() {
    let stdin = std::io::stdin();
    let mut source = LineSource::new(stdin.lock());
    input! {
        from &mut source,
        _: usize,
        w: usize,
        k: usize,
        _c: usize,
        water_sources: [Position; w],
        houses: [Position; k],
    }

    let querier = querier::Querier::new(source);
    let mut map = Map::new(querier);

    for h in &houses {
        let nearest_water_source = water_sources
            .iter()
            .min_by_key(|w| w.manhattan_distance(h))
            .unwrap();
        map.dig_until_break(nearest_water_source, 100);

        let diff = *nearest_water_source - *h;
        for dx in 0..diff.x.abs() {
            let i = h.x + dx * diff.x.signum();
            let j = h.y;
            map.dig_until_break(&Position::new(i, j), 100);
        }
        for dy in 0..diff.y.abs() {
            let i = h.x + diff.x;
            let j = h.y + dy * diff.y.signum();
            map.dig_until_break(&Position::new(i, j), 100);
        }
    }
}

mod querier {
    use super::{input, LineSource, Position};
    use std::io::StdinLock;
    use std::str::FromStr;

    enum Response {
        NotBroken = 0,
        Broken = 1,
        Finish = 2,
        Invalid = -1,
    }
    impl FromStr for Response {
        type Err = &'static str;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "0" => Ok(Response::NotBroken),
                "1" => Ok(Response::Broken),
                "2" => Ok(Response::Finish),
                "-1" => Ok(Response::Invalid),
                _ => Err("Unknown response"),
            }
        }
    }

    pub struct Querier<'a> {
        input_source: LineSource<StdinLock<'a>>,
    }
    impl<'a> std::fmt::Debug for Querier<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "Querier")
        }
    }
    impl<'a> Querier<'a> {
        pub fn new(input_source: LineSource<StdinLock<'a>>) -> Self {
            Self { input_source }
        }
        pub fn query(&mut self, position: &Position, power: i32) -> bool {
            println!("{} {} {}", position.x, position.y, power);
            input! {
                from &mut self.input_source,
                response: Response,
            }
            match response {
                Response::Broken => true,
                Response::NotBroken => false,
                Response::Finish => std::process::exit(0),
                Response::Invalid => std::process::exit(1),
            }
        }
    }
}

mod spatial_interpolater {
    use super::{Position, MAX_Z, MIN_Z, N};
    use itertools::Itertools;
    use nalgebra::{DMatrix, DVector};
    use std::ops::Range;

    pub struct SpatialInterpolator {
        samples: DMatrix<f64>,
        // A smoothing parameter that controls the spatial correlation of the kriging model
        alpha: f64,
        // The inverse of the kernel nxn matrix, used to compute the kriging weights
        k_inv: DMatrix<f64>,
        // Kriging weights
        z_weights: DVector<f64>,
    }
    impl std::fmt::Debug for SpatialInterpolator {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let prediction = self.predict_ranges(0..N as i32, 0..N as i32);
            let result = prediction.iter().map(|row| row.iter().join(",")).join("\n");
            writeln!(f, "{}", result)
        }
    }
    impl SpatialInterpolator {
        pub fn new(alpha: f64) -> Self {
            SpatialInterpolator {
                samples: DMatrix::zeros(0, 3),
                alpha,
                k_inv: DMatrix::zeros(0, 0),
                z_weights: DVector::zeros(0),
            }
        }
        pub fn train(&mut self, data: &[(&Position, i32)]) {
            let n = data.len();
            let mut samples = DMatrix::zeros(n, 3);

            for (i, (p, z)) in data.iter().enumerate() {
                samples[(i, 0)] = p.x as f64;
                samples[(i, 1)] = p.y as f64;
                samples[(i, 2)] = *z as f64;
            }

            let k = self.kernel_matrix(&samples);
            self.k_inv = (self.alpha * DMatrix::identity(n, n) + k)
                .try_inverse()
                .unwrap();
            self.z_weights = self.k_inv.clone() * samples.column(2);
            self.samples = samples;
        }
        #[allow(clippy::manual_clamp)]
        pub fn predict(&self, p: &Position) -> i32 {
            let n = self.samples.nrows();
            let mut k = DVector::zeros(n);
            for i in 0..n {
                let xi = self.samples[(i, 0)];
                let yi = self.samples[(i, 1)];
                k[i] = self.kernel(p.x as f64, p.y as f64, xi, yi);
            }

            let mut sum = 0.0;
            for i in 0..n {
                sum += k[i] * self.z_weights[i];
            }

            (sum as i32).max(MIN_Z).min(MAX_Z)
        }
        pub fn predict_ranges(&self, x_range: Range<i32>, y_range: Range<i32>) -> Vec<Vec<i32>> {
            let mut result = vec![];
            for i in x_range {
                let mut row = vec![];
                for j in y_range.clone() {
                    let p = Position::new(i, j);
                    let z = self.predict(&p);
                    row.push(z);
                }
                result.push(row);
            }
            result
        }
        fn kernel_matrix(&self, samples: &DMatrix<f64>) -> DMatrix<f64> {
            let n = samples.nrows();
            let mut k = DMatrix::zeros(n, n);

            for i in 0..n {
                let xi = samples[(i, 0)];
                let yi = samples[(i, 1)];
                let x_diff = samples.slice((0, 0), (n, 1)) - DMatrix::repeat(n, 1, xi);
                let y_diff = samples.slice((0, 1), (n, 1)) - DMatrix::repeat(n, 1, yi);
                let dist = (x_diff.component_mul(&x_diff) + y_diff.component_mul(&y_diff))
                    .map(|d| (-self.alpha * d).exp());
                k.column_mut(i).copy_from(&dist);
            }
            k
        }
        fn kernel(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
            let d2 = (x1 - x2).powi(2) + (y1 - y2).powi(2);
            (-self.alpha * d2).exp()
        }
    }
}
