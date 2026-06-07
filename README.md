# game_engine

A from-scratch 3D game engine in **C++** with **native (hardware) ray tracing**, built on
**Vulkan** and targeting Linux + AMD RDNA 3 (Radeon 780M).

The renderer is a **pure path tracer** (no rasterizer) using hardware-accelerated ray tracing via
the Vulkan KHR RT extensions (starting with inline `ray_query`), built as a wavefront path tracer
over a BLAS/TLAS acceleration structure. GPU kernels are **Vulkan compute shaders** (GLSL →
SPIR-V), not OpenCL/CUDA. Scene organization is an **ECS (EnTT)**. Because it's Vulkan, the engine
is cross-vendor — it also runs on NVIDIA/Intel, not just the AMD target.

See **[DESIGN.md](DESIGN.md)** — start with *Decisions at a glance* — for the full set of choices,
architecture, and the milestone plan.

## Status

Early design phase — no code yet. First milestone: a Vulkan "hello window".

## Quick check: is the RT stack available?

```bash
sudo pacman -S vulkan-tools vulkan-radeon
vulkaninfo | grep -iE 'ray_tracing|acceleration_structure|ray_query'
```
