mod cubes;
mod framework;
mod graphics;

fn main() {
    framework::run::<graphics::Cubes>("cube");
}
