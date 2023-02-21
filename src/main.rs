use proconio::{input, source::line::LineSource};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::collections::BTreeSet;

const N: usize = 200;
const MIN_Z: i32 = 10;
const MAX_Z: i32 = 5000;

fn main() {
    let stdin = std::io::stdin();
    let mut source = LineSource::new(stdin.lock());
    input! {
        from &mut source,
        _: usize,
        w: usize,
        k: usize,
        _c: usize,
        water_sources: [position::Position; w],
        houses: [position::Position; k],
    }

    let querier = querier::Querier::new(source);
    let mut map = map::Map::new(querier, &water_sources);

    let mut rng = Pcg64Mcg::new(42);
    let mut survey_positions = BTreeSet::new();
    houses.iter().for_each(|h| {
        survey_positions.insert(*h);
    });
    while survey_positions.len() < 100 {
        let i = rng.gen_range(0, N);
        let j = rng.gen_range(0, N);
        survey_positions.insert(position::Position::new(i, j));
    }
    survey_positions
        .iter()
        .for_each(|p| map.dig_until_break(p, 100));
    let surveryed_samples: Vec<_> = survey_positions
        .iter()
        .map(|p| (p, map.power_consumed(p)))
        .collect();
    let mut interpolater = spatial_interpolater::SpatialInterpolator::new(5e-3);
    interpolater.train(&surveryed_samples);
    let mut grid_costs = interpolater.predict_ranges(0..N, 0..N);

    for h in &houses {
        for (i, row) in grid_costs.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                let position = position::Position::new(i, j);
                let has_water = map.has_water(&position);
                if has_water {
                    *value = 0;
                }
            }
        }
        let min_cost_path = path_finder::calc_min_cost_path(&grid_costs, h, &water_sources);
        for p in min_cost_path {
            map.dig(&p, grid_costs[p.x][p.y]);
            map.dig_until_break(&p, 100);
        }
    }
}

mod position {
    use proconio::derive_readable;
    #[derive_readable]
    #[derive(Debug, PartialEq, Clone, Copy, Eq, PartialOrd, Ord)]
    pub struct Position {
        pub x: usize,
        pub y: usize,
    }
    impl Position {
        pub fn new(x: usize, y: usize) -> Self {
            Self { x, y }
        }
    }

    fn adjacent_grids(
        position: Position,
        height: usize,
        width: usize,
        directions: &[(usize, usize)],
    ) -> impl Iterator<Item = Position> + '_ {
        assert!(height < !0 && width < !0);
        directions.iter().filter_map(move |&(di, dj)| {
            let ni = (position.x).wrapping_add(di);
            let nj = (position.y).wrapping_add(dj);
            if ni < height && nj < width {
                Some(Position::new(ni, nj))
            } else {
                None
            }
        })
    }
    pub fn adjacent_grids_4(
        position: Position,
        height: usize,
        width: usize,
    ) -> impl Iterator<Item = Position> {
        adjacent_grids(position, height, width, &[(0, 1), (1, 0), (0, !0), (!0, 0)])
    }
}

mod map {
    use super::{
        dsu::DisjointSet,
        position::{adjacent_grids_4, Position},
        querier::Querier,
        N,
    };

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
    pub struct Map<'a> {
        querier: Querier<'a>,
        water_sources: Vec<Position>,
        water_dsu: DisjointSet,
        state: Vec<Vec<Pixel>>,
    }
    impl<'a> Map<'a> {
        pub fn new(querier: Querier<'a>, water_sources: &[Position]) -> Self {
            let water_sources = water_sources.to_vec();
            let water_dsu = DisjointSet::new(N, N);
            Self {
                querier,
                water_sources,
                water_dsu,
                state: vec![vec![Pixel::new(); N]; N],
            }
        }
        fn ref_pixel(&self, position: &Position) -> &Pixel {
            &self.state[position.x][position.y]
        }
        fn ref_mut_pixel(&mut self, position: &Position) -> &mut Pixel {
            &mut self.state[position.x][position.y]
        }
        pub fn is_broken(&self, position: &Position) -> bool {
            self.ref_pixel(position).is_broken
        }
        pub fn power_consumed(&self, position: &Position) -> i32 {
            self.ref_pixel(position).power_consumed
        }
        pub fn dig(&mut self, position: &Position, power: i32) -> bool {
            if self.is_broken(position) {
                return true;
            }
            let broke_after_dig = self.querier.query(position, power);

            let mut target_pixel = self.ref_mut_pixel(position);
            target_pixel.power_consumed += power;
            if broke_after_dig {
                target_pixel.is_broken = true;
                adjacent_grids_4(*position, N, N).for_each(|neighbor_p| {
                    if self.ref_pixel(&neighbor_p).is_broken {
                        self.water_dsu.merge_positions(&neighbor_p, position);
                    }
                });
            }
            broke_after_dig
        }
        pub fn dig_until_break(&mut self, position: &Position, power: i32) {
            loop {
                if self.dig(position, power) {
                    return;
                }
            }
        }
        pub fn has_water(&mut self, position: &Position) -> bool {
            if self.water_sources.iter().any(|ws| ws == position) {
                return self.is_broken(position);
            }
            (0..self.water_sources.len()).any(|i| {
                self.water_dsu
                    .same_positions(&self.water_sources[i], position)
            })
        }
    }
}

mod path_finder {
    use super::position::{adjacent_grids_4, Position};
    use std::{
        cmp::Reverse,
        collections::{BTreeMap, BTreeSet, BinaryHeap},
    };

    const INF: i32 = 1 << 30;

    pub fn calc_min_cost_path(
        cost: &Vec<Vec<i32>>,
        start: &Position,
        destinations: &[Position],
    ) -> Vec<Position> {
        let (min_cost, parent) = dijkstra(cost, start);
        let nearest_destination = destinations
            .iter()
            .min_by_key(|d| min_cost[d.x][d.y])
            .unwrap();
        reconstruct_path(&parent, start, nearest_destination)
    }

    fn dijkstra(
        grid_costs: &Vec<Vec<i32>>,
        start: &Position,
    ) -> (Vec<Vec<i32>>, BTreeMap<Position, Position>) {
        let height = grid_costs.len();
        let width = grid_costs[0].len();
        let mut result = vec![vec![INF; width]; height];
        let mut parent = BTreeMap::new();
        let mut heap = BinaryHeap::new();
        let mut visited = BTreeSet::new();

        result[start.x][start.y] = 0;
        heap.push(Reverse((0, *start)));

        while let Some(Reverse((cost, p))) = heap.pop() {
            if visited.contains(&p) {
                continue;
            }
            visited.insert(p);

            for np in adjacent_grids_4(p, height, width) {
                let ni = np.x;
                let nj = np.y;
                let next_cost = cost + grid_costs[ni][nj];
                if next_cost < result[ni][nj] {
                    result[ni][nj] = next_cost;
                    parent.insert(np, p);
                    heap.push(Reverse((next_cost, np)));
                }
            }
        }
        (result, parent)
    }

    fn reconstruct_path(
        parent: &BTreeMap<Position, Position>,
        start: &Position,
        destination: &Position,
    ) -> Vec<Position> {
        let mut path = vec![];
        let mut cur = *destination;
        while &cur != start {
            path.push(cur);
            cur = *parent.get(&cur).unwrap();
        }
        path.push(*start);
        path.reverse();
        path
    }
}

mod querier {
    use super::{input, position::Position, LineSource};
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
    use super::{position::Position, MAX_Z, MIN_Z, N};
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
            let prediction = self.predict_ranges(0..N, 0..N);
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
        pub fn predict_ranges(
            &self,
            x_range: Range<usize>,
            y_range: Range<usize>,
        ) -> Vec<Vec<i32>> {
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

mod dsu {
    use super::position::Position;

    #[derive(Debug)]
    pub struct DisjointSet {
        n: usize,
        height: usize,
        parent_or_size: Vec<i32>,
    }
    impl DisjointSet {
        pub fn new(height: usize, width: usize) -> Self {
            Self {
                n: height * width,
                height,
                parent_or_size: vec![-1; height * width],
            }
        }
        fn merge(&mut self, a: usize, b: usize) -> usize {
            assert!(a < self.n);
            assert!(b < self.n);
            let mut x = self.leader(a);
            let mut y = self.leader(b);
            if x == y {
                return x;
            }
            if -self.parent_or_size[x] < -self.parent_or_size[y] {
                std::mem::swap(&mut x, &mut y);
            }
            self.parent_or_size[x] += self.parent_or_size[y];
            self.parent_or_size[y] = x as i32;
            x
        }
        pub fn merge_positions(&mut self, a: &Position, b: &Position) -> Position {
            let a = self.encode(a);
            let b = self.encode(b);
            let parent = self.merge(a, b);
            self.decode(parent)
        }
        fn same(&mut self, a: usize, b: usize) -> bool {
            assert!(a < self.n);
            assert!(b < self.n);
            self.leader(a) == self.leader(b)
        }
        pub fn same_positions(&mut self, a: &Position, b: &Position) -> bool {
            let a = self.encode(a);
            let b = self.encode(b);
            self.same(a, b)
        }
        fn leader(&mut self, a: usize) -> usize {
            assert!(a < self.n);
            if self.parent_or_size[a] < 0 {
                a
            } else {
                self.parent_or_size[a] = self.leader(self.parent_or_size[a] as usize) as i32;
                self.parent_or_size[a] as usize
            }
        }
        fn encode(&self, position: &Position) -> usize {
            position.x * self.height + position.y
        }
        fn decode(&self, n: usize) -> Position {
            let x = n / self.height;
            let y = n % self.height;
            Position::new(x, y)
        }
    }
}
