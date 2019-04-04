use log::info;
use std::collections::VecDeque;

#[allow(dead_code)]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

pub fn load_glsl(name: &str, stage: ShaderStage) -> Vec<u8> {
    use std::fs::read_to_string;
    use std::io::Read;
    use std::path::PathBuf;

    let ty = match stage {
        ShaderStage::Vertex => glsl_to_spirv::ShaderType::Vertex,
        ShaderStage::Fragment => glsl_to_spirv::ShaderType::Fragment,
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(name);
    let code = match read_to_string(&path) {
        Ok(code) => code,
        Err(e) => panic!("Unable to read {:?}: {:?}", path, e),
    };

    let mut output = glsl_to_spirv::compile(&code, ty).unwrap();
    let mut spv = Vec::new();
    output.read_to_end(&mut spv).unwrap();
    spv
}

pub trait App {
    fn init(sc_desc: &wgpu::SwapChainDescriptor, device: &mut wgpu::Device) -> Self;
    fn resize(&mut self, sc_desc: &wgpu::SwapChainDescriptor, device: &mut wgpu::Device);
    fn tick(&mut self, delta: f32);
    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &mut wgpu::Device);
}

pub fn run<E: App>(title: &str) {
    use wgpu::winit::{
        ElementState, Event, EventsLoop, KeyboardInput, VirtualKeyCode, Window, WindowEvent,
    };

    info!("Initializing the device...");
    env_logger::init();
    let instance = wgpu::Instance::new();
    let adapter = instance.get_adapter(&wgpu::AdapterDescriptor {
        power_preference: wgpu::PowerPreference::LowPower,
    });
    let mut device = adapter.create_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
    });

    info!("Initializing the window...");
    let mut events_loop = EventsLoop::new();
    let window = Window::new(&events_loop).unwrap();
    window.set_title(title);
    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());

    let surface = instance.create_surface(&window);
    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsageFlags::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: size.width.round() as u32,
        height: size.height.round() as u32,
    };
    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    info!("Initializing the example...");
    let mut example = E::init(&sc_desc, &mut device);

    info!("Entering render loop...");
    let mut running = true;
    let mut moment = std::time::Instant::now();
    let mut fps_data = VecDeque::new();

    while running {
        let duration = moment.elapsed();
        let delta = duration.as_secs() as f32 + (duration.subsec_nanos() as f32 * 1.0e-9);
        if fps_data.len() > 1000 {
            fps_data.pop_front();
        }
        fps_data.push_back(delta);
        moment = std::time::Instant::now();
        events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                let physical = size.to_physical(window.get_hidpi_factor());
                info!("Resizing to {:?}", physical);
                sc_desc.width = physical.width.round() as u32;
                sc_desc.height = physical.height.round() as u32;
                swap_chain = device.create_swap_chain(&surface, &sc_desc);
                example.resize(&sc_desc, &mut device);
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | WindowEvent::CloseRequested => {
                    running = false;
                }
                _ => {}
            },
            _ => (),
        });
        example.tick(delta);

        let frame = swap_chain.get_next_texture();
        example.render(&frame, &mut device);
    }

    println!(
        "Moving average fps: {}",
        1.0 / (fps_data.iter().sum::<f32>() / fps_data.len() as f32)
    );
}
