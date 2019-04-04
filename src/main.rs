mod cubes;
mod framework;
mod app;

fn main() {
    framework::run::<app::Cubes>("cube");
}
