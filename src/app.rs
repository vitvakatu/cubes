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

pub struct Renderer {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    depth_view: wgpu::TextureView,
    instances: Vec<Instance>,
    index_count: usize,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    aspect_ratio: f32,
}

pub struct Cubes {
    renderer: Renderer,
    nodes: froggy::Storage<Node>,
    cubes: Vec<Cube>,
    levels: froggy::Storage<Level>,
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
        let mut cubes = create_cubes(&mut nodes, &levels, settings.scale);
        println!(
            "Initialized {} cubes on {} levels",
            cubes.len(),
            settings.levels_count
        );

        // Graphics
        let init_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        // Create the vertex and index buffers
        let vertex_size = mem::size_of::<Vertex>();
        let (vertex_data, index_data) = create_vertices();
        let vertex_buf = Cubes::create_vertex_buffer(device, &vertex_data);

        let index_buf = Cubes::create_index_buffer(device, &index_data);

        // Create instance buffer
        let instance_size = mem::size_of::<Instance>();
        let mut instances = Vec::new();
        Cubes::update_instances(&mut nodes, &mut cubes, &levels, &mut instances);
        let instance_buf = Cubes::create_instance_buffer(device, &instances);

        // Create pipeline layout
        let (bind_group_layout, pipeline_layout) = Cubes::create_pipeline_layout(device);

        let (aspect_ratio, uniform_buf) = Cubes::create_globals_uniform(sc_desc, device);

        // Create bind group
        let bind_group = Cubes::create_bind_group(device, &bind_group_layout, &uniform_buf);

        // Create the render pipeline
        let pipeline = Cubes::create_render_pipeline(
            &sc_desc,
            device,
            vertex_size,
            instance_size,
            &pipeline_layout,
        );

        let depth_texture = Cubes::create_depth_texture(&sc_desc, device);

        // Done
        let init_command_buf = init_encoder.finish();
        device.get_queue().submit(&[init_command_buf]);

        let renderer = Renderer {
            vertex_buf,
            index_buf,
            instance_buf,
            depth_view: depth_texture.create_default_view(),
            instances,
            index_count: index_data.len(),
            bind_group,
            uniform_buf,
            pipeline,
            aspect_ratio,
        };

        Cubes {
            nodes,
            cubes,
            levels,
            renderer,
            settings,
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

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        encoder.copy_buffer_to_buffer(&temp_buf, 0, &self.renderer.uniform_buf, 0, 64);
        device.get_queue().submit(&[encoder.finish()]);
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
        Self::update_instances(
            &self.nodes,
            &self.cubes,
            &self.levels,
            &mut self.renderer.instances,
        );
    }

    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &mut wgpu::Device) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        // load instances data
        let temp_buf = device
            .create_buffer_mapped(
                self.renderer.instances.len(),
                wgpu::BufferUsageFlags::TRANSFER_SRC,
            )
            .fill_from_slice(&self.renderer.instances);
        encoder.copy_buffer_to_buffer(
            &temp_buf,
            0,
            &self.renderer.instance_buf,
            0,
            (self.renderer.instances.len() * std::mem::size_of::<Instance>()) as u32,
        );
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
            rpass.set_bind_group(0, &self.renderer.bind_group);
            rpass.set_index_buffer(&self.renderer.index_buf, 0);
            rpass.set_vertex_buffers(&[
                (&self.renderer.vertex_buf, 0),
                (&self.renderer.instance_buf, 0),
            ]);
            rpass.draw_indexed(
                0..self.renderer.index_count as u32,
                0,
                0..self.renderer.instances.len() as u32,
            );
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
    ) -> wgpu::RenderPipeline {
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
        pipeline
    }

    fn create_pipeline_layout(
        device: &mut wgpu::Device,
    ) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
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
        (bind_group_layout, pipeline_layout)
    }

    fn create_globals_uniform(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &mut wgpu::Device,
    ) -> (f32, wgpu::Buffer) {
        let aspect_ratio = sc_desc.width as f32 / sc_desc.height as f32;
        let mx_total = Self::view_proj_matrix(aspect_ratio);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        let uniform_buf = device
            .create_buffer_mapped(
                16,
                wgpu::BufferUsageFlags::UNIFORM | wgpu::BufferUsageFlags::TRANSFER_DST,
            )
            .fill_from_slice(mx_ref);
        (aspect_ratio, uniform_buf)
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

    fn update_instances(
        nodes: &froggy::Storage<Node>,
        cubes: &[Cube],
        levels: &froggy::Storage<Level>,
        instances: &mut Vec<Instance>,
    ) {
        instances.clear();
        for cube in cubes {
            let space = nodes[&cube.node].world;
            let color = levels[&cube.level].color;
            instances.push(Instance {
                _offset_scale: space.disp.extend(space.scale).into(),
                _rotation: space.rot.v.extend(space.rot.s).into(),
                _color: color,
            });
        }
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
}
