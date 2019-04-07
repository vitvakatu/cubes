use crate::cubes::*;
use crate::framework;
use crate::Settings;
use cgmath::{Rotation3, Transform, Zero};

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
        vertex([-1, -1, 2], [0, 0, 1]),
        vertex([1, -1, 2], [0, 0, 1]),
        vertex([1, 1, 2], [0, 0, 1]),
        vertex([-1, 1, 2], [0, 0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, 0], [0, 0, -1]),
        vertex([1, 1, 0], [0, 0, -1]),
        vertex([1, -1, 0], [0, 0, -1]),
        vertex([-1, -1, 0], [0, 0, -1]),
        // right (1, 0, 0)
        vertex([1, -1, 0], [1, 0, 0]),
        vertex([1, 1, 0], [1, 0, 0]),
        vertex([1, 1, 2], [1, 0, 0]),
        vertex([1, -1, 2], [1, 0, 0]),
        // left (-1, 0, 0)
        vertex([-1, -1, 2], [-1, 0, 0]),
        vertex([-1, 1, 2], [-1, 0, 0]),
        vertex([-1, 1, 0], [-1, 0, 0]),
        vertex([-1, -1, 0], [-1, 0, 0]),
        // front (0, 1, 0)
        vertex([1, 1, 0], [0, 1, 0]),
        vertex([-1, 1, 0], [0, 1, 0]),
        vertex([-1, 1, 2], [0, 1, 0]),
        vertex([1, 1, 2], [0, 1, 0]),
        // back (0, -1, 0)
        vertex([1, -1, 2], [0, -1, 0]),
        vertex([-1, -1, 2], [0, -1, 0]),
        vertex([-1, -1, 0], [0, -1, 0]),
        vertex([1, -1, 0], [0, -1, 0]),
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
pub struct Instance {
    _offset_scale: [f32; 4],
    _rotation: [f32; 4],
    _color: [f32; 4],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct NoInstancingUniform {
    _projection: [f32; 16],
    _offset_scale: [f32; 4],
    _rotation: [f32; 4],
    _color: [f32; 4],
    // Padding for 256-bytes alignment
    _pad: [u8; 144],
}

impl NoInstancingUniform {
    pub fn new(space: Space, color: [f32; 4], projection_mx: [f32; 16]) -> Self {
        Self {
            _projection: projection_mx,
            _offset_scale: space.disp.extend(space.scale).into(),
            _rotation: space.rot.v.extend(space.rot.s).into(),
            _color: color,
            _pad: [0; 144],
        }
    }
}

pub enum InstancingMode {
    Instanced(InstancedMode),
    NonInstanced(NonInstancedMode),
}

pub struct InstancedMode {
    pub instances: Vec<Instance>,
    pub instance_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buf: wgpu::Buffer,
}

pub struct NonInstancedMode {
    pub uniforms_buf: wgpu::Buffer,
    pub uniforms: Vec<NoInstancingUniform>,
    pub bind_groups: Vec<wgpu::BindGroup>,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl InstancingMode {
    fn instanced(device: &mut wgpu::Device, aspect_ratio: f32, instances: Vec<Instance>) -> Self {
        let instance_buf = Cubes::create_instance_buffer(device, &instances);
        let uniform_buf = Cubes::create_globals_uniform(aspect_ratio, device);
        let bind_group_layout = Cubes::create_bind_group_layout(device);
        let bind_group = Cubes::create_bind_group(device, &bind_group_layout, &uniform_buf);
        InstancingMode::Instanced(InstancedMode {
            instance_buf,
            instances,
            bind_group,
            uniform_buf,
            bind_group_layout,
        })
    }

    fn non_instanced(device: &mut wgpu::Device, aspect_ratio: f32, world: &mut World) -> Self {
        let uniforms = Cubes::create_non_instancing_uniform_data(aspect_ratio, world);
        let uniforms_buf = Cubes::create_non_instanced_uniform_buffer(device, &uniforms);
        let bind_group_layout = Cubes::create_bind_group_layout(device);
        let bind_groups = Cubes::create_bind_groups_no_instancing(
            device,
            &bind_group_layout,
            &uniforms_buf,
            uniforms.len() as u32,
        );
        InstancingMode::NonInstanced(NonInstancedMode {
            uniforms_buf,
            uniforms,
            bind_groups,
            bind_group_layout,
        })
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        match *self {
            InstancingMode::Instanced(InstancedMode { ref bind_group_layout, .. }) => {
                bind_group_layout
            },
            InstancingMode::NonInstanced(NonInstancedMode { ref bind_group_layout, .. }) => {
                bind_group_layout
            },
        }
    }
}

pub struct Renderer {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    instancing_mode: InstancingMode,
    depth_view: wgpu::TextureView,
    index_count: usize,
    pipeline: wgpu::RenderPipeline,
    aspect_ratio: f32,
}

pub struct World {
    pub nodes: froggy::Storage<Node>,
    pub cubes: Vec<Cube>,
    pub levels: froggy::Storage<Level>,
}

pub struct Cubes {
    renderer: Renderer,
    world: World,
    settings: Settings,
    positions_was_updated: bool,
}

impl Cubes {
    fn view_proj_matrix(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_projection = cgmath::perspective(cgmath::Deg(60f32), aspect_ratio, 1.0, 100.0);
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(-1.8f32, -8.0, 4.0),
            cgmath::Point3::new(0f32, 0.0, 3.0),
            -cgmath::Vector3::unit_z(),
        );
        mx_projection * mx_view
    }
}

impl framework::App for Cubes {
    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &mut wgpu::Device,
        settings: Settings,
    ) -> Self {
        use std::mem;

        // Cubes
        // feed Froggy
        let mut nodes = froggy::Storage::new();
        let levels: froggy::Storage<_> =
            LEVELS.iter().cloned().take(settings.levels_count).collect();
        // Note: we populated the storages, but the returned pointers are already dropped.
        // Thus, all will be lost if we lock for writing now, but locking for reading retains the
        // contents, and cube creation will add references to them, so they will stay alive.
        let cubes = create_cubes(&mut nodes, &levels, settings.scale);
        println!(
            "Initialized {} cubes on {} levels",
            cubes.len(),
            settings.levels_count
        );

        let mut world = World {
            nodes,
            cubes,
            levels,
        };

        // Graphics
        let init_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        let aspect_ratio = sc_desc.width as f32 / sc_desc.height as f32;

        // Create the vertex and index buffers
        let vertex_size = mem::size_of::<Vertex>();
        let (vertex_data, index_data) = create_vertices();
        let vertex_buf = Cubes::create_vertex_buffer(device, &vertex_data);

        let index_buf = Cubes::create_index_buffer(device, &index_data);

        // Create instance buffer
        let instance_size = mem::size_of::<Instance>();
        let mut instances = Vec::new();
        let instancing_mode = if settings.no_instancing {
            InstancingMode::non_instanced(device, aspect_ratio, &mut world)
        } else {
            world.update_instances(&mut instances);
            InstancingMode::instanced(device, aspect_ratio, instances)
        };

        // Create pipeline layout
        let pipeline_layout = Cubes::create_pipeline_layout(device, instancing_mode.bind_group_layout());

        // Create the render pipeline
        let pipeline = Cubes::create_render_pipeline(
            &sc_desc,
            device,
            vertex_size,
            instance_size,
            &pipeline_layout,
            settings.no_instancing,
        );

        let depth_texture = Cubes::create_depth_texture(&sc_desc, device);

        // Done
        let init_command_buf = init_encoder.finish();
        device.get_queue().submit(&[init_command_buf]);

        let renderer = Renderer {
            vertex_buf,
            index_buf,
            instancing_mode,
            depth_view: depth_texture.create_default_view(),
            index_count: index_data.len(),
            pipeline,
            aspect_ratio,
        };

        Cubes {
            renderer,
            settings,
            world,
            positions_was_updated: false,
        }
    }

    fn resize(&mut self, sc_desc: &wgpu::SwapChainDescriptor, device: &mut wgpu::Device) {
        let depth_texture = Self::create_depth_texture(sc_desc, device);
        self.renderer.depth_view = depth_texture.create_default_view();

        let aspect_ratio = sc_desc.width as f32 / sc_desc.height as f32;
        self.renderer.aspect_ratio = aspect_ratio;
        let mx_total = Self::view_proj_matrix(aspect_ratio);
        let mx_ref: &[f32; 16] = mx_total.as_ref();

        let temp_buf = device
            .create_buffer_mapped(16, wgpu::BufferUsageFlags::TRANSFER_SRC)
            .fill_from_slice(mx_ref);

        match self.renderer.instancing_mode {
            InstancingMode::Instanced(InstancedMode {
                ref uniform_buf, ..
            }) => {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
                encoder.copy_buffer_to_buffer(&temp_buf, 0, &uniform_buf, 0, 64);
                device.get_queue().submit(&[encoder.finish()]);
            }
            _ => {}
        }
    }

    fn tick(&mut self, delta: f32) {
        if self.settings.dont_move {
            if !self.positions_was_updated {
                self.positions_was_updated = true;
            } else {
                return;
            }
        }
        // animate local spaces
        for cube in self.world.cubes.iter_mut() {
            let speed = self.world.levels[&cube.level].speed;
            let angle = cgmath::Rad(delta * speed);
            self.world.nodes[&cube.node].local.concat_self(&Space {
                disp: cgmath::Vector3::zero(),
                rot: cgmath::Quaternion::from_angle_z(angle),
                scale: 1.0,
            });
        }

        // re-compute world spaces, using streaming iteration
        {
            let mut cursor = self.world.nodes.cursor();
            while let Some((left, mut item, _)) = cursor.next() {
                item.world = match item.parent {
                    Some(ref parent) => left.get(parent).unwrap().world.concat(&item.local),
                    None => item.local,
                };
            }
        }

        // Update instances
        if let InstancingMode::Instanced(InstancedMode {
            ref mut instances, ..
        }) = self.renderer.instancing_mode
        {
            self.world.update_instances(instances);
        }
    }

    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &mut wgpu::Device) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        match self.renderer.instancing_mode {
            InstancingMode::Instanced(InstancedMode {
                ref instances,
                ref instance_buf,
                ..
            }) => {
                let temp_buf = device
                    .create_buffer_mapped(instances.len(), wgpu::BufferUsageFlags::TRANSFER_SRC)
                    .fill_from_slice(&instances);
                encoder.copy_buffer_to_buffer(
                    &temp_buf,
                    0,
                    &instance_buf,
                    0,
                    (instances.len() * std::mem::size_of::<Instance>()) as u32,
                );
            }
            InstancingMode::NonInstanced(NonInstancedMode {
                ref uniforms_buf, ..
            }) => {
                let mut uniforms_data = Vec::new();
                let mx = Self::view_proj_matrix(self.renderer.aspect_ratio);
                let mx_ref: &[f32; 16] = mx.as_ref();
                for cube in &self.world.cubes {
                    let space = self.world.nodes[&cube.node].world;
                    let color = self.world.levels[&cube.level].color;
                    let uniform_data = NoInstancingUniform::new(space, color, mx_ref.clone());
                    uniforms_data.push(uniform_data);
                }
                let temp_buf = device
                    .create_buffer_mapped(
                        uniforms_data.len(),
                        wgpu::BufferUsageFlags::UNIFORM | wgpu::BufferUsageFlags::TRANSFER_SRC,
                    )
                    .fill_from_slice(&uniforms_data);
                encoder.copy_buffer_to_buffer(
                    &temp_buf,
                    0,
                    &uniforms_buf,
                    0,
                    (uniforms_data.len() * std::mem::size_of::<NoInstancingUniform>()) as u32,
                );
            }
        }
        {
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.renderer.depth_view,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_stencil: 0,
                }),
            });
            rpass.set_pipeline(&self.renderer.pipeline);
            rpass.set_index_buffer(&self.renderer.index_buf, 0);
            match self.renderer.instancing_mode {
                InstancingMode::Instanced(InstancedMode {
                    ref bind_group,
                    ref instances,
                    ref instance_buf,
                    ..
                }) => {
                    rpass.set_vertex_buffers(&[(&instance_buf, 0), (&self.renderer.vertex_buf, 0)]);
                    rpass.set_bind_group(0, &bind_group);
                    rpass.draw_indexed(
                        0..self.renderer.index_count as u32,
                        0,
                        0..instances.len() as u32,
                    );
                }
                InstancingMode::NonInstanced(NonInstancedMode {
                    ref bind_groups, ..
                }) => {
                    rpass.set_vertex_buffers(&[(&self.renderer.vertex_buf, 0)]);
                    for i in 0..self.world.cubes.iter().count() {
                        rpass.set_bind_group(0, &bind_groups[i]);
                        rpass.draw_indexed(0..self.renderer.index_count as u32, 0, 0..1);
                    }
                }
            }
        }

        device.get_queue().submit(&[encoder.finish()]);
    }
}

impl Cubes {
    fn create_render_pipeline(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &mut wgpu::Device,
        vertex_size: usize,
        instance_size: usize,
        pipeline_layout: &wgpu::PipelineLayout,
        no_instancing: bool,
    ) -> wgpu::RenderPipeline {
        let vert_shader = if no_instancing {
            "cube_no_instancing.vert"
        } else {
            "cube.vert"
        };
        let vs_bytes = framework::load_glsl(vert_shader, framework::ShaderStage::Vertex);
        let fs_bytes = framework::load_glsl("cube.frag", framework::ShaderStage::Fragment);
        let vs_module = device.create_shader_module(&vs_bytes);
        let fs_module = device.create_shader_module(&fs_bytes);
        let vertex_buffer_descriptor = wgpu::VertexBufferDescriptor {
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
        };
        let instance_buffer_descriptor = wgpu::VertexBufferDescriptor {
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
        };
        let instancing_buffers = &[vertex_buffer_descriptor, instance_buffer_descriptor];
        let non_instancing_buffers = &instancing_buffers[0..1];
        let vertex_buffers = if no_instancing {
            non_instancing_buffers
        } else {
            instancing_buffers
        };
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
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: wgpu::TextureFormat::D32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers,
            sample_count: 1,
        });
        pipeline
    }

    fn create_pipeline_layout(device: &mut wgpu::Device, bind_group_layout: &wgpu::BindGroupLayout) -> wgpu::PipelineLayout {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[bind_group_layout],
        })
    }

    fn create_bind_group_layout(device: &mut wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStageFlags::VERTEX,
                ty: wgpu::BindingType::UniformBuffer,
            }],
        })
    }

    fn create_globals_uniform(aspect_ratio: f32, device: &mut wgpu::Device) -> wgpu::Buffer {
        let mx_total = Self::view_proj_matrix(aspect_ratio);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        device
            .create_buffer_mapped(
                16,
                wgpu::BufferUsageFlags::UNIFORM | wgpu::BufferUsageFlags::TRANSFER_DST,
            )
            .fill_from_slice(mx_ref)
    }

    fn create_bind_group(
        device: &mut wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &uniform_buf,
                    range: 0..64,
                },
            }],
        })
    }

    fn create_bind_groups_no_instancing(
        device: &mut wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        count: u32,
    ) -> Vec<wgpu::BindGroup> {
        let size = std::mem::size_of::<NoInstancingUniform>();
        let mut bind_groups = Vec::new();
        for i in 0..count {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                bindings: &[wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buf,
                        range: (i * size as u32)..(i + 1) * size as u32,
                    },
                }],
            });
            bind_groups.push(bind_group);
        }
        bind_groups
    }

    fn create_depth_texture(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &mut wgpu::Device,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: sc_desc.width,
                height: sc_desc.height,
                depth: 1,
            },
            array_size: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::D32Float,
            usage: wgpu::TextureUsageFlags::OUTPUT_ATTACHMENT,
        })
    }

    fn create_vertex_buffer(device: &mut wgpu::Device, vertex_data: &[Vertex]) -> wgpu::Buffer {
        device
            .create_buffer_mapped(vertex_data.len(), wgpu::BufferUsageFlags::VERTEX)
            .fill_from_slice(&vertex_data)
    }

    fn create_index_buffer(device: &mut wgpu::Device, index_data: &[u16]) -> wgpu::Buffer {
        device
            .create_buffer_mapped(index_data.len(), wgpu::BufferUsageFlags::INDEX)
            .fill_from_slice(&index_data)
    }

    fn create_instance_buffer(device: &mut wgpu::Device, instances: &[Instance]) -> wgpu::Buffer {
        device
            .create_buffer_mapped(
                instances.len(),
                wgpu::BufferUsageFlags::VERTEX | wgpu::BufferUsageFlags::TRANSFER_DST,
            )
            .fill_from_slice(&instances)
    }

    fn create_non_instanced_uniform_buffer(
        device: &mut wgpu::Device,
        uniforms_data: &[NoInstancingUniform],
    ) -> wgpu::Buffer {
        let no_instancing_buf = device
            .create_buffer_mapped(
                uniforms_data.len(),
                wgpu::BufferUsageFlags::UNIFORM | wgpu::BufferUsageFlags::TRANSFER_DST,
            )
            .fill_from_slice(&uniforms_data);
        no_instancing_buf
    }

    fn update_non_instancing_uniform_data(
        aspect_ratio: f32,
        world: &mut World,
        uniforms_data: &mut Vec<NoInstancingUniform>,
    ) {
        uniforms_data.clear();
        let mx = Self::view_proj_matrix(aspect_ratio);
        let mx_ref: &[f32; 16] = mx.as_ref();
        for cube in &world.cubes {
            let space = world.nodes[&cube.node].world;
            let color = world.levels[&cube.level].color;
            let uniform_data = NoInstancingUniform::new(space, color, mx_ref.clone());
            uniforms_data.push(uniform_data);
        }
    }

    fn create_non_instancing_uniform_data(
        aspect_ratio: f32,
        world: &mut World,
    ) -> Vec<NoInstancingUniform> {
        let mut uniforms_data = Vec::new();
        Self::update_non_instancing_uniform_data(aspect_ratio, world, &mut uniforms_data);
        uniforms_data
    }
}

impl World {
    pub fn update_instances(&self, instances: &mut Vec<Instance>) {
        instances.clear();
        for cube in self.cubes.iter() {
            let space = self.nodes[&cube.node].world;
            let color = self.levels[&cube.level].color;
            instances.push(Instance {
                _offset_scale: space.disp.extend(space.scale).into(),
                _rotation: space.rot.v.extend(space.rot.s).into(),
                _color: color,
            });
        }
    }
}
