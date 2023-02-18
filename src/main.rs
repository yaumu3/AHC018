use proconio::{derive_readable, input, source::line::LineSource};
use std::io::StdinLock;
use std::ops::{Add, Sub};
use std::str::FromStr;

const N: usize = 200;

#[derive_readable]
#[derive(Debug, PartialEq, Clone, Copy)]
struct Position {
    x: i32,
    y: i32,
}
impl Add for Position {
    type Output = Self;

    fn add(self, other: Position) -> Self {
        Position {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
impl Sub for Position {
    type Output = Position;

    fn sub(self, other: Position) -> Position {
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

struct Map<'a> {
    querier: Querier<'a>,
    state: [[bool; N]; N],
}
impl<'a> Map<'a> {
    fn new(querier: Querier<'a>) -> Self {
        Self {
            querier,
            state: [[false; N]; N],
        }
    }
    fn is_broken(&self, position: &Position) -> bool {
        self.state[position.x as usize][position.y as usize]
    }
    fn dig(&mut self, position: &Position, power: usize) -> bool {
        if self.is_broken(position) {
            return true;
        }
        let did_brake = self.querier.query(position, power);
        if did_brake {
            self.state[position.x as usize][position.y as usize] = true;
        }
        did_brake
    }
}

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
struct Querier<'a> {
    input_source: LineSource<StdinLock<'a>>,
}
impl<'a> Querier<'a> {
    fn new(input_source: LineSource<StdinLock<'a>>) -> Self {
        Self { input_source }
    }
    fn query(&mut self, position: &Position, power: usize) -> bool {
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

    let querier = Querier::new(source);
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
