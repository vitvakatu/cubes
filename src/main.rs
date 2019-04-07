mod app;
mod cubes;
mod framework;

use structopt::StructOpt;

#[derive(StructOpt, Debug, Clone, Copy)]
#[structopt(name = "cubes")]
pub struct Settings {
    #[structopt(
        short = "c",
        long = "count",
        default_value = "6",
        parse(try_from_str = "parse_cubes_count")
    )]
    levels_count: usize,
    #[structopt(short = "m", long = "dont-move")]
    dont_move: bool,
    #[structopt(short = "s", long = "scale", default_value = "0.4")]
    scale: f32,
    #[structopt(long = "no-instancing")]
    no_instancing: bool,
}

fn parse_cubes_count(input: &str) -> Result<usize, std::num::ParseIntError> {
    use crate::cubes::LEVELS;
    let mut cubes_count = input.parse::<usize>()?;
    if cubes_count < 1 {
        cubes_count = 1;
    } else if cubes_count > LEVELS.len() {
        cubes_count = LEVELS.len();
    }
    Ok(cubes_count)
}

fn main() {
    let settings = Settings::from_args();
    framework::run::<app::Cubes>("cube", settings);
}
