use proconio::{derive_readable, input, source::line::LineSource};

const N: usize = 200;

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

        loop {
            if map.dig(nearest_water_source, 100) {
                break;
            }
        }

        let diff = *nearest_water_source - *h;
        for dx in 0..diff.x.abs() {
            let i = h.x + dx * diff.x.signum();
            let j = h.y;
            loop {
                if map.dig(&Position::new(i, j), 100) {
                    break;
                }
            }
        }
        for dy in 0..diff.y.abs() {
            let i = h.x + diff.x;
            let j = h.y + dy * diff.y.signum();
            loop {
                if map.dig(&Position::new(i, j), 100) {
                    break;
                }
            }
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
