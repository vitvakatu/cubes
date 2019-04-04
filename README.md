# WGPU-cubes
Cubes example from https://github.com/kvark/froggy, written on top of wgpu-rs

# How to run

Use the following command to run the demo with default settings:

```bash
cargo run --release --features <backend>
```

Available backends are:
- `metal`
- `vulkan`
- `dx12`
- `dx11`

The demo also has additional settings, you can print help by adding `--help` argument:

```bash
cargo run --release --features backend -- --help
```
