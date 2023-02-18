use proconio::{derive_readable, input, source::line::LineSource};
use std::ops::{Add, Sub};
use std::str::FromStr;

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
    fn manhattan_distance(&self, other: &Position) -> i32 {
        let diff = *self - *other;
        diff.x.abs() + diff.y.abs()
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

fn main() {
    let stdin = std::io::stdin();
    let mut source = LineSource::new(stdin.lock());
    input! {
        from &mut source,
        n: usize,
        w: usize,
        k: usize,
        c: usize,
        water_sources: [Position; w],
        houses: [Position; k],
    }
    for i in 0..n {
        for j in 0..n {
            println!("{} {} {}", i, j, 5000);
            input! {
                from &mut source,
                response: Response,
            };
            if let Response::Finish = response {
                return;
            }
        }
    }
}
