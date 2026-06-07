# Game Engine — Design Document

A from-scratch 3D game engine in **C++** with **native (hardware) ray tracing**, built and
run on Linux. This document records the goals, key technical decisions, architecture, and a
milestone plan.

---

## Decisions at a glance (confirmed)

| Area | Decision | Rationale |
|------|----------|-----------|
| Language | **C++** | Industry standard for engines; max control & learning value |
| Dimensionality | **3D** | — |
| Graphics + compute API | **Vulkan** | Only cross-platform API exposing hardware RT; one API for graphics + compute + present. Portable — also runs on NVIDIA/Intel, so not AMD lock-in. |
| GPU kernels | **Vulkan compute shaders** (GLSL → SPIR-V) — *not* OpenCL or CUDA | Only path that can both invoke hardware ray tracing *and* share resources with rendering/presentation. OpenCL/CUDA are peer compute APIs that can't do RT or present. |
| Ray tracing | **Native (hardware)** via `VK_KHR` RT; start with inline `ray_query` | Drives the RDNA3 Ray Accelerators; `ray_query` is the simpler entry point before the full RT pipeline |
| Renderer | **Pure path tracer** (no rasterizer) | Single clean path and a true RT showcase; accepts heavier reliance on denoising/temporal on the iGPU |
| Scene model | **ECS via EnTT** — designed & owned by Arthur | Data-oriented; pairs naturally with GPU buffers. Renderer reads from it via a clean interface. |
| RHI | **Thin Vulkan wrapper that evolves** (single backend) | Tame boilerplate without speculative multi-backend architecture; extract more only when a second use appears |
| Denoising/temporal | **Architect for it early, deliver in stages** | Reserve buffers/frame-graph from the first RT milestone; implement basic PT → temporal accumulation → spatial denoiser in that order |
| Engine↔renderer API | **Extract to neutral types; zero-copy buffer fill; snapshot** | Engine writes neutral `InstanceDesc` straight into a renderer-provided mapped buffer (true zero-copy on the shared-memory iGPU); renderer converts to Vulkan format on-GPU. Engine has no Vulkan, renderer has no EnTT. (§5) |
| Target HW | Linux · AMD RDNA3 Radeon 780M (RADV) | Has hardware RT; iGPU → modest perf, so dev/learning focus |

**Code split:** host code (C++/Vulkan) orchestrates — resources, BLAS/TLAS builds, dispatch,
sync, present; device code (GLSL compute shaders) *is* the path tracer — ray gen, bounce loop
with `rayQuery`, shading, sampling, accumulation.

**Now resolved (see §3.5, §4.3):** RHI → **thin Vulkan wrapper that evolves**; denoising/temporal
→ **architected-for early, delivered in stages** (basic PT → accumulation → denoiser).

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
- **Portability bonus:** Vulkan is cross-vendor. The same KHR-RT code runs on NVIDIA (RT cores)
  and Intel (RTUs) as well as our AMD Ray Accelerators — choosing Vulkan is the *opposite* of
  vendor lock-in, and the engine will run (likely faster) on a discrete NVIDIA/AMD card.

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
  stack as our RT — one toolchain for both compute kernels and ray tracing.
- **Our GPU kernels are Vulkan compute shaders** (GLSL → SPIR-V), *not* OpenCL/CUDA kernels.
  "GPU kernel" is a generic concept; OpenCL, CUDA, and Vulkan compute shaders are all ways to
  write one (compute world says "kernel", graphics world says "shader" — same thing). We use
  compute shaders because they are the only kernels that can invoke `rayQuery` (hardware RT) and
  natively share buffers/images with the renderer and swapchain. OpenCL is a *peer* API (same
  stack level, neither calls the other) that is compute-only — it can't trace rays or present —
  so we don't use it. (OpenCL/SYCL remain available only as fallbacks for non-RT general compute.)

### 3.4 Rendering model: **pure path tracer**, separable from the engine
- The renderer is the *only* subsystem that fundamentally cares about RT. Scene management, ECS,
  asset loading, input, audio, physics are identical regardless of the rendering method.
- **Decision: pure path tracing — no rasterizer.** *Everything*, including primary camera
  visibility, is ray-traced. Cleaner single code path and a true RT showcase.
- **Implications we accept:**
  - The TLAS must cover **all visible geometry every frame** (primary visibility uses it too) —
    the acceleration-structure build is on the critical path for everything.
  - **Noise is the main enemy on the 780M.** With few samples-per-pixel the raw image is grainy,
    so **temporal accumulation + denoising carry the real-time burden** and are promoted toward
    first-class (see open question on timing in the Decisions summary).
  - (A hybrid raster+RT path is *not* planned; if real-time perf on the iGPU forces it, it can be
    reconsidered, but the design target is pure PT.)

### 3.5 RHI: **thin Vulkan wrapper that evolves** (single backend)
- Build a *minimal* abstraction — `Device`, `Buffer`, `Image`, `Pipeline`, `CommandList`, plus
  acceleration-structure / `rayQuery` helpers — sized to keep path-tracer code readable.
- **Keep it Vulkan-shaped and single-backend.** Wrap the boilerplate, but do *not* hide concepts
  we genuinely need to control (barriers, memory types, sync). No speculative DX12/Metal layer.
- **Extract new abstractions only when a second real use appears** — matches "build it, then
  extract the engine." Avoids both raw-Vulkan boilerplate sprawl in the renderer and up-front
  architecture for backends that don't exist.

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

#### Denoising/temporal plan: architect early, deliver in stages
This is structurally invasive, so we **reserve the architecture from the first RT milestone**
even though we implement it later:
- **Reserve up front:** render into an offscreen **HDR accumulation target with ping-pong
  history**, and have the path tracer emit **auxiliary buffers from the primary hit** (depth,
  normal, albedo, motion vectors). Both temporal reprojection and SVGF-style denoising need
  these; adding them early is cheap, retrofitting is painful.
- **Deliver in order:** (1) basic *noisy* path tracer first (need an image before denoising it);
  (2) **temporal accumulation** next (highest value / lowest cost — a static-camera accumulator
  converges immediately); (3) a **spatial denoiser** (À-Trous / SVGF-style) as a self-contained
  pass. No off-the-shelf vendor denoiser (e.g. NVIDIA OptiX) on AMD, so this is our own / an open
  implementation.

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

## 5. Engine ↔ Renderer API contract

The most important internal boundary. **Engine core never touches Vulkan; renderer never touches
EnTT.** The only vocabulary between them is a small set of neutral POD types plus a registration +
per-frame submission API.

**Interaction model: extract.** Each frame the engine walks its EnTT registry and writes a
renderer-neutral snapshot; the renderer consumes it. The renderer includes no EnTT headers; the
engine includes no Vulkan types.

### 5.1 Layer 1 — Resource registration (infrequent: on asset load)
Geometry/materials/textures are registered **once**; the renderer uploads them to the GPU and
builds a **BLAS** per mesh, returning an opaque handle (the "build once" half). The vertex/index
buffers serve **both** the BLAS build and in-shader surface fetch.

```cpp
struct MeshHandle     { uint32_t id; };
struct MaterialHandle { uint32_t id; };
struct TextureHandle  { uint32_t id; };   // id==0 == none

struct Vertex { vec3 pos; vec3 normal; vec2 uv; vec4 tangent; };

struct MeshDesc {
    std::span<const Vertex>   vertices;
    std::span<const uint32_t> indices;
    bool deformable = false;          // true → fast-build BLAS, refit via updateMesh
};
struct MaterialDesc {
    vec3 baseColor{1,1,1}; float metallic=0, roughness=1; vec3 emission{0,0,0};
    TextureHandle albedo{}, normalMap{}, metalRough{};
};
struct TextureDesc { const void* pixels; uint32_t w, h; Format fmt; };

MeshHandle     registerMesh(const MeshDesc&);
MaterialHandle registerMaterial(const MaterialDesc&);
TextureHandle  registerTexture(const TextureDesc&);
void           updateMesh(MeshHandle, std::span<const Vertex>);   // deformables → BLAS refit
void           release(MeshHandle /* + overloads */);
```

### 5.2 Layer 2 — Per-frame submission (every frame: zero-copy fill)
**Snapshot style** (not delta): the engine submits the full instance set each frame and the
renderer rebuilds the cheap TLAS. Delta only saves *scene-description* cost at large scale and
adds stateful complexity — the path tracer re-traces the whole frame regardless (camera motion +
temporal accumulation + global illumination mean there is no "unchanged region" to skip). The door
stays open to delta later as a pure submission-layer optimization.

**Zero-copy buffer fill:** the renderer hands the engine a writable span into its own
**persistently-mapped, double-buffered** instance buffer; the engine's extraction writes straight
in — no engine-side array, no copy.

**iGPU bonus:** on the 780M, host-visible memory *is* device-local (shared memory), so this is
**true zero-copy** — the CPU writes the same physical memory the GPU reads, with no staging
transfer (which a discrete card would require).

**Format conversion stays on-GPU:** the engine writes neutral `InstanceDesc`; a renderer compute
pass expands these into `VkAccelerationStructureInstanceKHR` + per-instance shading records for the
TLAS build — so the Vulkan instance format never crosses the boundary.

```cpp
struct CameraDesc { vec3 position; quat orientation; float vFovRad, aspect, nearZ, farZ; };

struct InstanceDesc {              // renderer-defined POD (no Vulkan types)
    MeshHandle mesh; MaterialHandle material;
    mat4 transform;                // world transform THIS frame
    uint64_t stableId;             // persistent per-object id → motion vectors
};
struct LightDesc { enum Type {Directional,Point,Spot} type; vec3 dir,pos,color; float intensity,radius; };
struct EnvironmentDesc { TextureHandle envMap; float intensity; vec3 tint; };

// Per-frame, zero-copy:
FrameBuilder fb = renderer.beginFrame();          // picks the right double-buffer
std::span<InstanceDesc> dst = fb.instances(maxN); // mapped GPU-visible memory
size_t n = 0; /* engine ECS extraction writes dst[n++] = {...} */
fb.setInstanceCount(n);
fb.setCamera(cam); fb.setLights(lights); fb.setEnvironment(env);
fb.setFrameIndex(idx); fb.setResetAccumulation(cameraCut);
renderer.submit(fb, target);
```

### 5.3 Cross-cutting contracts
- **Vulkan ownership:** entirely the renderer. Only CPU-side data + opaque handles cross over.
- **Motion vectors:** the renderer tracks previous transforms keyed by `InstanceDesc::stableId`.
  The engine only keeps `stableId` stable and sets `resetAccumulation` on camera cuts.
- **Static vs deformable:** `MeshDesc::deformable` sets BLAS build flags; `updateMesh()` refits
  deformables; **rigid motion needs neither** — just a new `transform` in `InstanceDesc`
  (TLAS-only update).
- **Write-combined memory:** the engine writes the instance span **linearly, forward, no
  read-back**.
- **Double-buffering** (frames in flight) is the renderer's job, hidden behind `beginFrame()`.
- **Lifetime:** the `FrameBuilder` and its spans are valid until `submit()`; that call is the
  synchronization point.

---

## 6. Engine subsystems (beyond the renderer)

Sequenced roughly by when they're needed:

1. **Platform layer** — window + input (GLFW or SDL2).
2. **Game loop** — fixed-timestep update, interpolated render.
3. **Math** — vectors/matrices/quaternions (GLM initially).
4. **Renderer** — Vulkan core, then the RT path above.
5. **Asset loading** — images (stb_image), 3D models (glTF/Assimp).
6. **Debug UI** — Dear ImGui (added early; basis for a future editor).
7. **Scene organization** — **ECS via EnTT, designed & owned by Arthur.** The engine core exposes
   the ECS; the renderer only *reads* it each frame. Contract: iterate entities with
   `{ mesh handle, material, world transform }` → renderer packs them into GPU buffers + TLAS
   instances. The renderer is built to this interface and stays out of the ECS design.
8. **Physics/collision** — integrate Jolt or Bullet later.
9. **Audio** — a library (OpenAL / miniaudio) later.
10. **Serialization & scripting** — later (e.g. Lua via sol2).

---

## 7. Dependencies & tooling

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

## 8. Milestone ladder

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

## 9. Open questions / decisions deferred
- glTF (hand-parse to understand the data) vs. Assimp (convenience) for model loading.
- When to migrate from `ray_query` to the full RT pipeline.
- HLSL vs. GLSL for shader authoring (default GLSL).

### Resolved this session
- Graphics + compute API → **Vulkan** (cross-vendor; the one stack for graphics + compute + RT).
- GPU kernels → **Vulkan compute shaders**, not OpenCL/CUDA.
- Rendering model → **pure path tracer** (no rasterizer).
- Scene model → **ECS via EnTT**, owned by Arthur.
- RHI → **thin Vulkan wrapper that evolves**, single backend (§3.5).
- Denoising/temporal → **architected-for early, delivered in stages** (§4.3).
- Engine↔renderer API → **extract model, zero-copy buffer fill, snapshot submission**; motion
  vectors tracked renderer-side via `stableId` (§5).
