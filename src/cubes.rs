use cgmath::{Angle, One, Rotation3, Transform, Zero};

pub type Space = cgmath::Decomposed<cgmath::Vector3<f32>, cgmath::Quaternion<f32>>;

pub struct Node {
    pub local: Space,
    pub world: Space,
    pub parent: Option<froggy::Pointer<Node>>,
}

#[derive(Clone, Copy)]
pub struct Level {
    pub speed: f32,
    pub color: [f32; 4],
}

pub struct Cube {
    pub node: froggy::Pointer<Node>,
    pub level: froggy::Pointer<Level>,
}

pub const LEVELS: [Level; 6] = [
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
    Level {
        speed: -1.6,
        color: [1.0, 0.5, 0.5, 1.0],
    },
    Level {
        speed: 1.9,
        color: [0.5, 1.0, 1.0, 1.0],
    },
    Level {
        speed: -2.2,
        color: [1.0, 0.5, 1.0, 1.0],
    },
];

pub fn create_cubes(
    nodes: &mut froggy::Storage<Node>,
    levels: &froggy::Storage<Level>,
) -> Vec<Cube> {
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
