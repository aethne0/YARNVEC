//!  ▄· ▄▌ ▄▄▄· ▄▄▄   ▐ ▄  ▌ ▐·▄▄▄ . ▄▄· 
//! ▐█▪██▌▐█ ▀█ ▀▄ █·•█▌▐█▪█·█▌▀▄.▀·▐█ ▌▪
//! ▐█▌▐█▪▄█▀▀█ ▐▀▀▄ ▐█▐▐▌▐█▐█•▐▀▀▪▄██ ▄▄
//!  ▐█▀·.▐█ ▪▐▌▐█•█▌██▐█▌ ███ ▐█▄▄▌▐███▌
//!   ▀ •  ▀  ▀ .▀  ▀▀▀ █▪. ▀   ▀▀▀ ·▀▀▀ 
//!
//! **YARNVEC** is a vector/matrix/game math SIMD library
//!
//! *Maintainer*:   github.com/aethne0 
//! *Version*:      0.0.1
//! *Date*:         2026-03-02
//! *License*:      Apache | MIT
//!
//! https://github.com/aethne0/YARNVEC
//! Please make an issue for any bugs, performance optimizations, or 
//! if you can point to a faster implementation of any math functions.

pub const vec = @import("vector.zig");
pub const mat = @import("matrix.zig");

fn VectorDef(comptime dims: usize) type {
    if (dims == 0) @compileError("dims must be 1 or greater");

    return extern struct {
        const Self = @This();

        x: f32,
        y: if (dims > 1) f32 else void = {},
        z: if (dims > 1) f32 else void = {},
        w: if (dims > 1) f32 else void = {},
    };
}

