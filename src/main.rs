mod app;
mod cubes;
mod framework;

fn main() {
    framework::run::<app::Cubes>("cube");
}
