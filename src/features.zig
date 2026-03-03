const std = @import("std");
const builtin = @import("builtin");

pub const X86_SSE = builtin.cpu.features.isEnabled(@intFromEnum(std.Target.x86.Feature.sse));
pub const X86_AVX = builtin.cpu.features.isEnabled(@intFromEnum(std.Target.x86.Feature.avx));
pub const AARCH64_NEON = builtin.cpu.features.isEnabled(@intFromEnum(std.Target.aarch64.Feature.neon));

pub extern fn zig_rsqrt_ps(@Vector(4, f32)) @Vector(4, f32);
