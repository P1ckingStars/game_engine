# Game Engine — Design Document

A from-scratch 3D game engine in **C++** with **native (hardware) ray tracing**, built and
run on Linux. This document records the goals, key technical decisions, architecture, and a
milestone plan.

---

## 1. Goals & scope

- **Language:** C++ (industry standard for engine work; maximum control and learning value).
- **Dimensionality:** 3D.
- **Headline feature:** *native* ray tracing — i.e. **hardware-accelerated** RT that drives the
  GPU's dedicated RT units, not a CPU/software ray tracer.
- **Guiding principle:** **build a game first, then extract the engine from it.** Make a tiny
  game work as a hard-coded prototype, then refactor the reusable parts out. Abstractions that
  survive a second game are the right ones. This avoids over-engineering an engine for problems
  no real game has.

### Non-goals (for now)
- Cross-platform parity from day one (Linux/AMD first; portability is a later concern).
- A full editor/tooling suite up front (add an ImGui-based debug UI early, a real editor late).
- Shipping-grade performance; correctness and understanding come first.

---

## 2. Target platform & hardware

- **OS:** Linux (Arch).
- **GPU:** AMD **Radeon 780M** (integrated), **RDNA 3** architecture, in a Ryzen 7 7840HS APU.
  - RDNA 3 includes **2nd-gen Ray Accelerators** — the fixed-function BVH-traversal / ray-box /
    ray-triangle hardware (AMD's equivalent of NVIDIA RT cores). **Hardware RT is supported.**
- **Driver stack:** `amdgpu` (kernel) + **RADV** (Mesa Vulkan user-mode driver). RADV exposes the
  Vulkan KHR ray tracing extensions on this GPU.

### Hardware constraints (accepted)
- It is an **integrated GPU**: 12 compute units / 12 Ray Accelerators, sharing system RAM (no
  dedicated VRAM, lower bandwidth). Real hardware RT works and is *correct*, but throughput is
  modest. **Target: development and learning at moderate resolution / samples-per-pixel**, not
  AAA real-time path tracing.

---

## 3. Key technical decisions (with rationale)

### 3.1 Graphics API: **Vulkan** (not OpenGL)
- Hardware ray tracing requires a modern explicit API. **OpenGL has no hardware RT path and never
  will** — it is a dead end for this project's headline feature.
- Options for native RT: **Vulkan KHR ray tracing** (cross-platform), DX12/DXR (Windows-only),
  Metal (Apple-only). For a from-scratch Linux engine, **Vulkan** is the choice.
- Consequence: we accept Vulkan's steep boilerplate up front rather than building on OpenGL and
  rewriting the renderer later.

### 3.2 RT entry point: **ray queries first**, full RT pipeline later
- **Ray queries (inline RT, `VK_KHR_ray_query`)**: trace hardware rays from inside an ordinary
  compute shader, with our own bounce loop — *no* shader binding table (SBT), *no* separate
  raygen/hit/miss stages. Far simpler first milestone.
- **Full RT pipeline (`VK_KHR_ray_tracing_pipeline`)**: raygen / closest-hit / any-hit / miss
  shaders orchestrated by the driver + SBT. Adopt later when the inline model is limiting.

### 3.3 Compute API: **Vulkan compute** (not CUDA, not ROCm/HIP)
- **CUDA is NVIDIA-only** — it does not run on this AMD GPU.
- **HIP/ROCm** (AMD's CUDA-equivalent) does not officially support this RDNA 3 iGPU (`gfx1103`);
  only fragile env-override hacks. Not a foundation to build on.
- **Vulkan compute** is the most reliable compute path on this iGPU on Linux **and** is the same
  stack as our RT — one toolchain for both compute kernels and ray tracing. (OpenCL and SYCL are
  available fallbacks for general compute if needed.)

### 3.4 Rendering model: **hybrid**, separable from the engine
- The renderer is the *only* subsystem that fundamentally cares about RT. Scene management, ECS,
  asset loading, input, audio, physics are identical whether we rasterize or ray trace.
- Plan for **hybrid rendering**: rasterize the main scene, use RT for the expensive-to-fake
  effects (reflections, shadows, ambient occlusion, global illumination).

---

## 4. Ray tracing architecture (the renderer core)

### 4.1 The acceleration structure: BVH via BLAS/TLAS
- A **BVH** (Bounding Volume Hierarchy) is a tree of nested axis-aligned bounding boxes (AABBs)
  over scene geometry. A ray walks it top-down, skipping any subtree whose box it misses, turning
  "test millions of triangles" into `~O(log n)` box tests. It is the structure the Ray
  Accelerators traverse.
- **Two-level structure (built/managed by the driver):**
  - **BLAS** (Bottom-Level): a BVH over one mesh's triangles. Built once; reused across frames and
    instances.
  - **TLAS** (Top-Level): a BVH over instances (each = a BLAS + a transform). Small; cheap to
    rebuild per frame.
- **Build/update strategy (build once, query by every ray):**
  - Static geometry → build once at load, reuse every frame.
  - Rigid motion → reuse BLAS, update only the instance transform and rebuild the small TLAS.
  - Deformation → refit the BLAS (keep topology, re-expand boxes; `O(n)`), full rebuild
    occasionally.
- The structure is **read-only during tracing**, so millions of ray threads query it lock-free.
  Clean split: **build = write phase, trace = read phase.**

### 4.2 What the RT hardware/API does vs. what we write
- **Hardware/API does ONLY the tree query:** traversal + intersection → returns a hit record
  (distance `t`, primitive index, barycentrics, instance id) or miss. It is a *hit-finding
  subroutine*, not a renderer.
- **We write everything else** in GPU kernels: primary ray generation, surface fetch (index our
  own vertex/material buffers from the primitive index), shading/BRDF, next-direction sampling,
  throughput, the bounce loop, accumulation, final pixel write.

### 4.3 Path tracing structure
- **Wavefront / streaming** path tracing: each bounce is its own full-GPU pass. Between passes,
  **stream-compact** surviving rays into a dense buffer (and optionally **sort by material**) to
  fight SIMT divergence and register pressure. Buffers in → buffers out (ray, throughput, hit,
  accumulation).
- **One pass = one full-GPU job**; parallelism is across the millions of rays in a bounce, *not*
  along any single ray's path.
- **Keep the GPU saturated** by spending spare occupancy on **more samples-per-pixel of the
  current frame** (and **path regeneration**: refill dead path slots with fresh primary rays),
  rather than pipelining future frames — improves the current image at zero added latency.
- **Bounce depth is capped** (e.g. 5) for termination + cost; Russian roulette for unbiased
  early termination later.
- **Temporal reuse** (accumulation / reprojection / ReSTIR) and **denoising** to get a clean
  image from few samples. **Async compute** to overlap heterogeneous passes (e.g. denoise on
  compute/tensor-style units while traversal runs on RT units).

### 4.4 Why dedicated RT hardware is required (design rationale)
- BVH traversal is branchy, pointer-chasing, **divergent** work: rays in a warp take different
  tree paths / depths → SIMT lockstep wastes lanes. This *intra-traversal* divergence cannot be
  compacted away (compaction works *between* passes; traversal divergence lives *inside* one
  trace, and secondary rays are directionally incoherent after a diffuse bounce).
- Filling the GPU fixes **occupancy**, not the **per-ray cost** of divergent traversal. Software
  RT can be 100% utilized and still far slower. Ray Accelerators attack the cost axis: not SIMT
  lanes, so divergent tree-walks are their native workload; they also offload traversal from the
  shader cores so shading overlaps.

---

## 5. Engine subsystems (beyond the renderer)

Sequenced roughly by when they're needed:

1. **Platform layer** — window + input (GLFW or SDL2).
2. **Game loop** — fixed-timestep update, interpolated render.
3. **Math** — vectors/matrices/quaternions (GLM initially).
4. **Renderer** — Vulkan core, then the RT path above.
5. **Asset loading** — images (stb_image), 3D models (glTF/Assimp).
6. **Debug UI** — Dear ImGui (added early; basis for a future editor).
7. **Scene organization** — ECS (EnTT, or a hand-rolled one to learn).
8. **Physics/collision** — integrate Jolt or Bullet later.
9. **Audio** — a library (OpenAL / miniaudio) later.
10. **Serialization & scripting** — later (e.g. Lua via sol2).

---

## 6. Dependencies & tooling

- **Build:** CMake (+ vcpkg or system packages for deps).
- **Vulkan:** Vulkan SDK / loader, validation layers, RADV driver (`vulkan-radeon`),
  `vulkan-tools` (for `vulkaninfo`).
- **Libraries:** GLFW or SDL2, GLM, stb_image, Dear ImGui, EnTT (later), Assimp/glTF, Jolt/Bullet
  (later).
- **Shaders:** GLSL → SPIR-V via `glslangValidator` / `glslc` (or HLSL if preferred later).

### Verify the RT stack on this machine
```bash
sudo pacman -S vulkan-tools vulkan-radeon
vulkaninfo | grep -iE 'ray_tracing|acceleration_structure|ray_query'
vulkaninfo | grep -iE 'deviceName|driverName'
```

---

## 7. Milestone ladder

Rendering foundation (the LearnOpenGL/vulkan-tutorial curriculum, in Vulkan):

1. **Hello window** — GLFW/SDL + Vulkan instance/device/swapchain; clear to a color.
   (Validates the whole CMake/build/link setup — historically the hardest first hurdle.)
2. **First triangle** — vertex buffer, vertex+fragment shaders, one draw.
3. **Textured quad** — stb_image, sampling.
4. **Transforms + camera** — GLM MVP, a spinning textured cube, perspective camera → *now 3D*.
5. **Camera controls + depth** — fly camera, depth buffer.
6. **Load a real model** — glTF/Assimp mesh with textures.
7. **Lighting** — Blinn-Phong, then toward PBR.

Ray tracing:

8. **Trace one hardware ray** — build a BLAS/TLAS, write a `ray_query` compute shader that
   shoots a primary ray and writes a color. *First native RT milestone.*
9. **Inline RT shading** — surface fetch + simple shading + shadow rays.
10. **Path tracing loop** — bounce loop, accumulation over samples, a noisy→clean image.
11. **Wavefront restructure** — separate passes + stream compaction.
12. **Hybrid renderer** — rasterize main pass, RT for reflections/shadows/AO/GI.
13. **Temporal accumulation + denoising.**

Engine extraction:

14. **Extract the engine** — Renderer/Mesh/Shader/Camera/Transform/ResourceManager, then ECS,
    then the rest of §5.

---

## 8. Open questions / decisions deferred
- ECS vs. traditional scene-graph for the first engine (lean ECS / EnTT).
- glTF (hand-parse to understand the data) vs. Assimp (convenience) for model loading.
- When to migrate from ray queries to the full RT pipeline.
- HLSL vs. GLSL for shader authoring.
- When (if) to add a second backend (DX12/Vulkan portability layer).
