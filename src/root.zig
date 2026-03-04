//!  ```
//!  ▄· ▄▌• ▌ ▄ ·.  ▄▄▄· ▄▄▄▄▄ ▄ .▄
//! ▐█▪██▌·██ ▐███▪▐█ ▀█ •██  ██▪▐█
//! ▐█▌▐█▪▐█ ▌▐▌▐█·▄█▀▀█  ▐█.▪██▀▐█
//!  ▐█▀·.██ ██▌▐█▌▐█ ▪▐▌ ▐█▌·██▌▐▀
//!   ▀ • ▀▀  █▪▀▀▀ ▀  ▀  ▀▀▀ ▀▀▀ ·
//!
//! **YMATH** is a vector/matrix/game math SIMD library
//!
//! *Maintainer*:   github.com/aethne0 
//! *Version*:      0.0.1
//! *Date*:         2026-03-02
//! *License*:      Apache
//!
//! https://codeberg.org/yarnf/ymath
//! Please make an issue for any bugs, performance optimizations, or 
//! if you can point to a faster implementation of any math functions.
//!   ```

comptime {
   @setFloatMode(.optimized);
}

const vec2 = @import("vector_2.zig");
/// 2-dimensional f32 vector
pub const Vec2  = vec2.Vector2(f32);
/// 2-dimensional f64 vector
pub const Vec2f64  = vec2.Vector2(f64);

const vec3_a = @import("vector_3a.zig");
/// aligned 3-dimensional f32 vector
pub const Vec3A  = vec3_a.Vec3A;
/// aligned 3-dimensional f64 vector
pub const Vec3Af64  = vec3_a.Vec3Af64;

const vec4 = @import("vector_4.zig");
/// 4-dimensional f32 vector
pub const Vec4  = vec4.Vector4(f32);
/// 4-dimensional f64 vector
pub const Vec4f64  = vec4.Vector4(f64);

const mat4 = @import("matrix4.zig");
/// 4x4 f32 matrix
pub const Mat4 = mat4.Mat4(f32);
/// 4x4 f64 matrix
pub const Mat4f64 = mat4.Mat4(f64);

const std = @import("std");
test { 
    std.testing.refAllDecls(@This()); 
}

