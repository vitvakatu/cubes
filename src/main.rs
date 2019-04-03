use cgmath::{Transform, Rotation3, Zero, One};

mod framework;

type Space = cgmath::Decomposed<cgmath::Vector3<f32>, cgmath::Quaternion<f32>>;

struct Node {
    local: Space,
    world: Space,
    parent: Option<froggy::Pointer<Node>>
}

struct Cube {
    node: froggy::Pointer<Node>,
}

#[derive(Clone, Copy)]
struct Vertex {
    _pos: [f32; 4],
    _normal: [i32; 4],
}

fn vertex(pos: [i8; 3], normal: [i8; 3]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _normal: [normal[0] as i32, normal[1] as i32, normal[2] as i32, 1]
    }
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0, 1]),
        vertex([1, -1, 1], [0, 0, 1]),
        vertex([1, 1, 1], [0, 0, 1]),
        vertex([-1, 1, 1], [0, 0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [0, 0, -1]),
        vertex([1, 1, -1], [0, 0, -1]),
        vertex([1, -1, -1], [0, 0, -1]),
        vertex([-1, -1, -1], [0, 0, -1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [1, 0, 0]),
        vertex([1, 1, -1], [1, 0, 0]),
        vertex([1, 1, 1], [1, 0, 0]),
        vertex([1, -1, 1], [1, 0, 0]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [-1, 0, 0]),
        vertex([-1, 1, 1], [-1, 0, 0]),
        vertex([-1, 1, -1], [-1, 0, 0]),
        vertex([-1, -1, -1], [-1, 0, 0]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [0, 1, 0]),
        vertex([-1, 1, -1], [0, 1, 0]),
        vertex([-1, 1, 1], [0, 1, 0]),
        vertex([1, 1, 1], [0, 1, 0]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, -1, 0]),
        vertex([-1, -1, 1], [0, -1, 0]),
        vertex([-1, -1, -1], [0, -1, 0]),
        vertex([1, -1, -1], [0, -1, 0]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

struct Example {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: usize,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    nodes: froggy::Storage<Node>,
    cubes: Vec<Cube>,
    aspect_ratio: f32,
}

impl Example {
    fn view_proj_matrix(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(1.5f32, -5.0, 3.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            -cgmath::Vector3::unit_z(),
        );
        mx_projection * mx_view
    }

    fn transform_matrix(global: Space, view_proj: cgmath::Matrix4<f32>) -> cgmath::Matrix4<f32> {
        view_proj * cgmath::Matrix4::from(global)
    }
}

impl framework::Example for Example {
    fn init(sc_desc: &wgpu::SwapChainDescriptor, device: &mut wgpu::Device) -> Self {
        use std::mem;

        let init_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        // Create the vertex and index buffers
        let vertex_size = mem::size_of::<Vertex>();
        let (vertex_data, index_data) = create_vertices();
        let vertex_buf = device
            .create_buffer_mapped(vertex_data.len(), wgpu::BufferUsageFlags::VERTEX)
            .fill_from_slice(&vertex_data);

        let index_buf = device
            .create_buffer_mapped(index_data.len(), wgpu::BufferUsageFlags::INDEX)
            .fill_from_slice(&index_data);

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStageFlags::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let aspect_ratio = sc_desc.width as f32 / sc_desc.height as f32;
        let mx_total = Self::view_proj_matrix(sc_desc.width as f32 / sc_desc.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        let uniform_buf = device
            .create_buffer_mapped(
                16,
                wgpu::BufferUsageFlags::UNIFORM | wgpu::BufferUsageFlags::TRANSFER_DST,
            )
            .fill_from_slice(mx_ref);

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buf,
                        range: 0..64,
                    },
                },
            ],
        });

        // Create the render pipeline
        let vs_bytes = framework::load_glsl("cube.vert", framework::ShaderStage::Vertex);
        let fs_bytes = framework::load_glsl("cube.frag", framework::ShaderStage::Fragment);
        let vs_module = device.create_shader_module(&vs_bytes);
        let fs_module = device.create_shader_module(&fs_bytes);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::PipelineStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: wgpu::PipelineStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            },
            rasterization_state: wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Cw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            },
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: sc_desc.format,
                color: wgpu::BlendDescriptor::REPLACE,
                alpha: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWriteFlags::ALL,
            }],
            depth_stencil_state: None,
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: vertex_size as u32,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttributeDescriptor {
                        attribute_index: 0,
                        format: wgpu::VertexFormat::Float4,
                        offset: 0,
                    },
                    wgpu::VertexAttributeDescriptor {
                        attribute_index: 1,
                        format: wgpu::VertexFormat::Int4,
                        offset: 4 * 4,
                    },
                ],
            }],
            sample_count: 1,
        });

        // Done
        let init_command_buf = init_encoder.finish();
        device.get_queue().submit(&[init_command_buf]);

        // Cubes
        let mut nodes = froggy::Storage::new();
        let node = nodes.create(Node {
            local: Space {
                scale: 1.0,
                rot: cgmath::Quaternion::one(),
                disp: cgmath::Vector3::zero(),
            },
            world: Space::one(),
            parent: None,
        });
        let cubes = vec![Cube {
            node,
        }];

        Example {
            vertex_buf,
            index_buf,
            index_count: index_data.len(),
            bind_group,
            uniform_buf,
            pipeline,
            nodes,
            cubes,
            aspect_ratio
        }
    }

    fn resize(&mut self, sc_desc: &wgpu::SwapChainDescriptor, device: &mut wgpu::Device) {
        let aspect_ratio = sc_desc.width as f32 / sc_desc.height as f32;
        self.aspect_ratio = aspect_ratio;
        let mx_total = Self::view_proj_matrix(aspect_ratio);
        let mx_ref: &[f32; 16] = mx_total.as_ref();

        let temp_buf = device
            .create_buffer_mapped(16, wgpu::BufferUsageFlags::TRANSFER_SRC)
            .fill_from_slice(mx_ref);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        encoder.copy_buffer_to_buffer(&temp_buf, 0, &self.uniform_buf, 0, 64);
        device.get_queue().submit(&[encoder.finish()]);
    }

    fn update(&mut self, _event: wgpu::winit::WindowEvent) {
        //empty
    }

    fn tick(&mut self, delta: f32) {
        // animate local spaces
        for cube in self.cubes.iter_mut() {
            let angle = cgmath::Rad(delta * 1.0);
            self.nodes[&cube.node].local.concat_self(&Space {
                disp: cgmath::Vector3::zero(),
                rot: cgmath::Quaternion::from_angle_z(angle),
                scale: 1.0,
            });
        }

        // re-compute world spaces, using streaming iteration
        {
            let mut cursor = self.nodes.cursor();
            while let Some((left, mut item, _)) = cursor.next() {
                item.world = match item.parent {
                    Some(ref parent) => left.get(parent).unwrap().world.concat(&item.local),
                    None => item.local,
                };
            }
        }
    }

    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &mut wgpu::Device) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        for cube in &self.cubes {
            let global = self.nodes[&cube.node].world;
            let matrix = Self::transform_matrix(global, Self::view_proj_matrix(self.aspect_ratio));
            let mx_ref: &[f32; 16] = matrix.as_ref();
            let temp_buf = device.create_buffer_mapped(16, wgpu::BufferUsageFlags::TRANSFER_SRC).fill_from_slice(mx_ref);
            encoder.copy_buffer_to_buffer(&temp_buf, 0, &self.uniform_buf, 0, 64);

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    },
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group);
            rpass.set_index_buffer(&self.index_buf, 0);
            rpass.set_vertex_buffers(&[(&self.vertex_buf, 0)]);
            rpass.draw_indexed(0..self.index_count as u32, 0, 0..1);
        }

        device.get_queue().submit(&[encoder.finish()]);
    }
}

fn main() {
    framework::run::<Example>("cube");
}
