# game_engine

A from-scratch 3D game engine in **C++** with **native (hardware) ray tracing**, built on
**Vulkan** and targeting Linux + AMD RDNA 3 (Radeon 780M).

The headline feature is hardware-accelerated ray tracing via the Vulkan KHR ray tracing
extensions (starting with inline `ray_query`), with the renderer built as a wavefront path
tracer over a BLAS/TLAS acceleration structure.

See **[DESIGN.md](DESIGN.md)** for goals, technical decisions, architecture, and the milestone
plan.

## Status

Early design phase — no code yet. First milestone: a Vulkan "hello window".

## Quick check: is the RT stack available?

```bash
sudo pacman -S vulkan-tools vulkan-radeon
vulkaninfo | grep -iE 'ray_tracing|acceleration_structure|ray_query'
```
