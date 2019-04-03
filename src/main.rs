use cgmath::{One, Rotation3, Transform, Zero, Angle};

mod framework;

type Space = cgmath::Decomposed<cgmath::Vector3<f32>, cgmath::Quaternion<f32>>;

struct Node {
    local: Space,
    world: Space,
    parent: Option<froggy::Pointer<Node>>,
}

#[derive(Clone, Copy)]
struct Level {
    speed: f32,
    color: [f32; 4],
}

struct Cube {
    node: froggy::Pointer<Node>,
    level: froggy::Pointer<Level>,
}

const LEVELS: [Level; 3] = [
    Level {
        speed: 0.7,
        color: [1.0, 1.0, 0.5, 1.0],
    },
    Level {
        speed: -1.0,
        color: [0.5, 0.5, 1.0, 1.0],
    },
    Level {
        speed: 1.3,
        color: [0.5, 1.0, 0.5, 1.0],
    },
//    Level {
//        speed: -1.6,
//        color: [1.0, 0.5, 0.5, 1.0],
//    },
//    Level {
//        speed: 1.9,
//        color: [0.5, 1.0, 1.0, 1.0],
//    },
//    Level {
//        speed: -2.2,
//        color: [1.0, 0.5, 1.0, 1.0],
//    },
];

fn create_cubes(nodes: &mut froggy::Storage<Node>, levels: &froggy::Storage<Level>) -> Vec<Cube> {
    let mut levels_iter = levels.iter_all();
    let root_level = levels_iter.next().unwrap();
    let mut list = vec![Cube {
        node: nodes.create(Node {
            local: Space {
                disp: cgmath::Vector3::zero(),
                rot: cgmath::Quaternion::one(),
                scale: 2.0,
            },
            world: Space::one(),
            parent: None,
        }),
        level: levels.pin(&root_level),
    }];
    struct Stack<'a> {
        parent: froggy::Pointer<Node>,
        levels_iter: froggy::Iter<'a, Level>,
    }
    let mut stack = vec![Stack {
        parent: list[0].node.clone(),
        levels_iter,
    }];

    let axis = [
        cgmath::Vector3::unit_z(),
        cgmath::Vector3::unit_x(),
        -cgmath::Vector3::unit_x(),
        cgmath::Vector3::unit_y(),
        -cgmath::Vector3::unit_y(),
    ];
    let children: Vec<_> = axis
        .iter()
        .map(|&axe| {
            Space {
                disp: cgmath::vec3(0.0, 0.0, 1.0),
                rot: cgmath::Quaternion::from_axis_angle(axe, cgmath::Rad::turn_div_4()),
                scale: 1.0,
            }
            .concat(&Space {
                disp: cgmath::vec3(0.0, 0.0, 1.0),
                rot: cgmath::Quaternion::one(),
                scale: 0.4,
            })
        })
        .collect();

    while let Some(mut next) = stack.pop() {
        //HACK: materials are indexed the same way as levels
        // it's fine for demostration purposes
        let level = match next.levels_iter.next() {
            Some(item) => levels.pin(&item),
            None => continue,
        };
        for child in &children {
            let cube = Cube {
                node: nodes.create(Node {
                    local: child.clone(),
                    world: Space::one(),
                    parent: Some(next.parent.clone()),
                }),
                level: level.clone(),
            };
            stack.push(Stack {
                parent: cube.node.clone(),
                levels_iter: next.levels_iter.clone(),
            });
            list.push(cube);
        }
    }

    list
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Vertex {
    _pos: [f32; 4],
    _normal: [i32; 4],
}

fn vertex(pos: [i8; 3], normal: [i8; 3]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _normal: [normal[0] as i32, normal[1] as i32, normal[2] as i32, 1],
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

#[derive(Clone, Copy)]
#[repr(C)]
struct Instance {
    _offset_scale: [f32; 4],
    _rotation: [f32; 4],
    _color: [f32; 4],
}

impl Instance {
    pub fn new(offset_scale: [f32; 4], rotation: cgmath::Quaternion<f32>, color: [f32; 4]) -> Self {
        Self {
            _offset_scale: offset_scale,
            _rotation: rotation.into(),
            _color: color,
        }
    }
}

struct Example {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    instances: Vec<Instance>,
    index_count: usize,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    nodes: froggy::Storage<Node>,
    cubes: Vec<Cube>,
    levels: froggy::Storage<Level>,
    aspect_ratio: f32,
}

impl Example {
    fn view_proj_matrix(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(-1.8f32, -8.0, 3.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            -cgmath::Vector3::unit_z(),
        );
        mx_projection * mx_view
    }
}

impl framework::Example for Example {
    fn init(sc_desc: &wgpu::SwapChainDescriptor, device: &mut wgpu::Device) -> Self {
        use std::mem;

        // Cubes
        // feed Froggy
        let mut nodes = froggy::Storage::new();
        let levels: froggy::Storage<_> = LEVELS.iter().cloned().collect();
        //Note: we populated the storages, but the returned pointers are already dropped.
        // Thus, all will be lost if we lock for writing now, but locking for reading retains the
        // contents, and cube creation will add references to them, so they will stay alive.
        let mut cubes = create_cubes(&mut nodes, &levels);
        println!(
            "Initialized {} cubes on {} levels",
            cubes.len(),
            LEVELS.len()
        );

        // Graphics

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

        // Create instance buffer
        let instance_size = mem::size_of::<Instance>();
        let mut instances = Vec::new();
        for cube in &cubes {
            let space = nodes[&cube.node].world;
            instances.push(Instance {
                _offset_scale: space.disp.extend(space.scale).into(),
                _rotation: space.rot.v.extend(space.rot.s).into(),
                _color: [1.0, 1.0, 1.0, 1.0],
            });
        }
        let instance_buf = device
            .create_buffer_mapped(
                instances.len(),
                wgpu::BufferUsageFlags::VERTEX | wgpu::BufferUsageFlags::TRANSFER_DST,
            )
            .fill_from_slice(&instances);

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStageFlags::VERTEX,
                ty: wgpu::BindingType::UniformBuffer,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let aspect_ratio = sc_desc.width as f32 / sc_desc.height as f32;
        let mx_total = Self::view_proj_matrix(aspect_ratio);
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
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &uniform_buf,
                    range: 0..64,
                },
            }],
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
            vertex_buffers: &[
                wgpu::VertexBufferDescriptor {
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
                },
                wgpu::VertexBufferDescriptor {
                    stride: instance_size as u32,
                    step_mode: wgpu::InputStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            attribute_index: 2,
                            format: wgpu::VertexFormat::Float4,
                            offset: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            attribute_index: 3,
                            format: wgpu::VertexFormat::Float4,
                            offset: 4 * 4,
                        },
                        wgpu::VertexAttributeDescriptor {
                            attribute_index: 4,
                            format: wgpu::VertexFormat::Float4,
                            offset: 4 * 4 * 2,
                        },
                    ],
                },
            ],
            sample_count: 1,
        });

        // Done
        let init_command_buf = init_encoder.finish();
        device.get_queue().submit(&[init_command_buf]);

        Example {
            vertex_buf,
            index_buf,
            instance_buf,
            instances,
            index_count: index_data.len(),
            bind_group,
            uniform_buf,
            pipeline,
            nodes,
            cubes,
            levels,
            aspect_ratio,
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
            let speed = self.levels[&cube.level].speed;
            let angle = cgmath::Rad(delta * speed);
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

        // Update instances
        self.instances.clear();
        for cube in &self.cubes {
            let space = self.nodes[&cube.node].world;
            let color = self.levels[&cube.level].color;
            self.instances.push(Instance {
                _offset_scale: space.disp.extend(space.scale).into(),
                _rotation: space.rot.v.extend(space.rot.s).into(),
                _color: color,
            });
        }
    }

    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &mut wgpu::Device) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        for cube in &self.cubes {
            // load instances data
            let temp_buf = device
                .create_buffer_mapped(self.instances.len(), wgpu::BufferUsageFlags::TRANSFER_SRC)
                .fill_from_slice(&self.instances);
            encoder.copy_buffer_to_buffer(
                &temp_buf,
                0,
                &self.instance_buf,
                0,
                (self.instances.len() * std::mem::size_of::<Instance>()) as u32,
            );

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
            rpass.set_vertex_buffers(&[(&self.vertex_buf, 0), (&self.instance_buf, 0)]);
            rpass.draw_indexed(
                0..self.index_count as u32,
                0,
                0..self.instances.len() as u32,
            );
        }

        device.get_queue().submit(&[encoder.finish()]);
    }
}

fn main() {
    framework::run::<Example>("cube");
}
