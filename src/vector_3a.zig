const std = @import("std");
const features = @import("features.zig");
const vec2 = @import("vector_2.zig");
const vec4 = @import("vector_4.zig");

pub fn Vector3A(comptime FType: type) type {
    if (FType != f32 and FType != f64 and FType != f128) 
        @compileError("fType must be f32, f64, f128");

    return extern struct {
        const Self = @This();

        x:      FType align(@sizeOf(FType) * 4),
        y:      FType,
        z:      FType,
        _pad:   FType = 0,

        /// Initialize vector with value for each dimension
        /// Example: `const some_vec = Vec3A.init(1, 2, 3);`
        pub fn init(x: FType, y: FType, z: FType) Self {
            return .{ .x = x, .y = y, .z = z };
        }

        /// Initializes vector with `scalar` for all elements
        /// Example `Vec3A.splat(2)` -> `{ 2, 2, 2 }`
        pub inline fn splat(scalar: FType) Self {
            var result: @Vector(4, FType) = @splat(scalar);
            result[3] = 0;
            return @bitCast(result);
        }

        /// Vector initialized with all values 0
        pub const ZERO      = splat(0);
        /// Vector initialized with all values 1
        pub const ONE       = splat(1);
        /// Vector initialized with all values -1
        pub const NEG_ONE   = splat(-1);
        /// X-direction unit vector
        pub const X         = init(1, 0, 0);
        /// Y-direction unit vector
        pub const Y         = init(0, 1, 0);
        /// Z-direction unit vector
        pub const Z         = init(0, 0, 1);
        /// negative X-direction unit vector
        pub const NEG_X     = init(-1, 0, 0);
        /// negative Y-direction unit vector
        pub const NEG_Y     = init(0, -1, 0);
        /// negative Z-direction unit vector
        pub const NEG_Z     = init(0, 0, -1);
        /// Vector initialized with min value of float type
        pub const MIN       = splat(std.math.floatMin(FType));
        /// Vector initialized with max value of float type
        pub const MAX       = splat(std.math.floatMax(FType));
        /// Vector initialized with NaN
        pub const NAN       = splat(std.math.nan(FType));
        /// Vector initialized with inf
        pub const INF       = splat(std.math.inf(FType));
        /// Vector initialized with negative inf
        pub const NEG_INF   = splat(-std.math.inf(FType));
        /// array of all positive unit vectors
        pub const AXES      = [_]Self{ X, Y, Z };

        inline fn as_vec(self: Self) @Vector(4, FType) {
            return @bitCast(self);
        }

        // This is to safely handle the pad element in some reduce operations (product, min, max)
        inline fn as_vec_3(self: Self) @Vector(3, FType) {
            return @shuffle(FType, self.as_vec(), undefined, @Vector(3, FType){0, 1, 2});
        }

        /// Element-wise bitwise equality
        pub fn eq(self: Self, other: Self) bool {
            std.debug.assert(self._pad == 0);
            std.debug.assert(other._pad == 0);

            // according to my experiments this emits faster asm than `@Reduce(.And, self == other)`
            // it just uses wide xor which is lower latency than the packed float compare
            // features.c.

            switch (FType) {
                f32  => return @as(u128, @bitCast(self.as_vec())) == @as(u128, @bitCast(other.as_vec())),
                f64  => return @as(u256, @bitCast(self.as_vec())) == @as(u256, @bitCast(other.as_vec())),
                f128 => return @as(u512, @bitCast(self.as_vec())) == @as(u512, @bitCast(other.as_vec())),
                else => unreachable,
            }
        }

        /// Element-wise add
        pub fn add(self: Self, other: Self) Self {
            return @bitCast( self.as_vec() + other.as_vec() );
        }

        /// Element-wise subtract
        pub fn sub(self: Self, other: Self) Self {
            return @bitCast( self.as_vec() - other.as_vec() );
        }

        /// Element-wise multiply
        pub fn mul(self: Self, other: Self) Self {
            return @bitCast(self.as_vec() * other.as_vec());
        }

        /// Element-wise divide
        pub fn div(self: Self, other: Self) Self {
            var result: Self = @bitCast(self.as_vec() / other.as_vec());
            result._pad = 0;
            return result;
        }

        /// Element-wise (self * multiplier) + addend. 
        /// Will use fused multiply add optimizations if available.
        pub fn mul_add(self: Self, multiplier: Self, addend: Self) Self {
            return @bitCast(
                @mulAdd(@Vector(4, FType),
                    self.as_vec(),
                    multiplier.as_vec(),
                    addend.as_vec()
                )
            );
        }

        /// Adds scalar to each element
        pub fn add_scalar(self: Self, scalar: FType) Self {
            return self.add(splat(scalar));
        }

        /// Subtracts scalar from each element
        pub fn sub_scalar(self: Self, scalar: FType) Self {
            return self.sub(splat(scalar));
        }

        /// Multiplies each element by scalar
        pub fn mul_scalar(self: Self, scalar: FType) Self {
            return self.mul(splat(scalar));
        }

        /// Divides each element by scalar
        pub fn div_scalar(self: Self, scalar: FType) Self {
            return self.div(splat(scalar));
        }

        /// Element-wise negate
        pub fn neg(self: Self) Self {
            return self.mul_scalar(-1);
        }

        /// Sum of all elements
        pub fn sum(self: Self) FType {
            std.debug.assert(self._pad == 0);
            return @reduce(.Add, self.as_vec());
        }

        /// Product of all elements
        pub fn product(self: Self) FType {
            return @reduce(.Mul, self.as_vec_3());
        }

        /// min of all elements
        pub fn min_element(self: Self) FType {
            return @reduce(.Min, self.as_vec_3());
        }

        /// max of all elements
        pub fn max_element(self: Self) FType {
            return @reduce(.Max, self.as_vec_3());
        }

        /// Element-wise ceil
        pub fn ceil(self: Self) Self {
            return @bitCast(@ceil(self.as_vec()));
        }

        /// Element-wise round
        pub fn round(self: Self) Self {
            return @bitCast(@round(self.as_vec()));
        }

        /// Element-wise floor
        pub fn floor(self: Self) Self {
            return @bitCast(@floor(self.as_vec()));
        }

        /// Element-wise sin
        pub fn sin(self: Self) FType {
            return @bitCast(@sin(self.as_vec()));
        }

        /// Element-wise cos
        pub fn cos(self: Self) FType {
            return @bitCast(@cos(self.as_vec()));
        }

        /// Element-wise natural logarithm
        pub fn ln(self: Self) FType {
            return @bitCast(@log(self.as_vec()));
        }

        /// Element-wise base-2 logarithm
        pub fn log2(self: Self) FType {
            return @bitCast(@log2(self.as_vec()));
        }

        /// Element-wise e^self
        pub fn exp(self: Self) FType {
            return @bitCast(@exp(self.as_vec()));
        }

        /// Element-wise 2^self
        pub fn exp2(self: Self) FType {
            return @bitCast(@exp2(self.as_vec()));
        }

        /// Element-wise reciprocal (1/x)
        pub fn recip(self: Self) FType {
            return ONE.div(self);
        }

        /// Element-wise sqrt
        pub fn sqrt(self: Self) FType {
            return @bitCast(@sqrt(self.as_vec()));
        }

        /// Approximate reciprocal square root of each element, only faster on arch that supports it,
        /// and usually only for f32.
        pub fn recip_sqrt_fast(self: Self) Self {
            // todo: perf
            // this doesnt seem to emit anything very good for aarch64 (f32, probably not f64 either)
            // https://godbolt.org/z/9x4vjfozn
            // with: -O ReleaseFast -target aarch64-linux-gnu -mcpu cortex_a55
            //      fast_recip_root:
            //          stp     x29, x30, [sp, #-16]!
            //          mov     x29, sp
            //          fsqrt   v0.4s, v0.4s
            //          fmov    v1.4s, #1.00000000
            //          fdiv    v0.4s, v1.4s, v0.4s
            //          ldp     x29, x30, [sp], #16
            //          ret
            // we should be using `FRSQRTE Vd.4S,Vn.4S` and `FRSQRTS Vd.4S,Vn.4S,Vm.4S`
            //
            // for non-f32 sizes we have nothing else to do
            //
            // For x86 this seems to use vrsqrtps so big need to mess around.
            @setFloatMode(.optimized);
            const one: @Vector(4, FType) = @splat(1);
            return @bitCast(one / @sqrt(self.as_vec()));
        }

        /// Element-wise absolute value
        pub fn abs(self: Self) Self {
            return @bitCast(@abs(self.as_vec()));
        }

        /// Computes dot product with another vector
        pub fn dot(self: Self, other: Self) FType {
            std.debug.assert(self._pad == 0);
            std.debug.assert(other._pad == 0);
            return self.mul(other).sum();
        }

        /// Element-wise max operation
        pub fn max(self: Self, other: Self) Self {
            return @bitCast(@max(self.as_vec(), other.as_vec()));
        }

        /// Element-wise min operation
        pub fn min(self: Self, other: Self) Self {
            return @bitCast(@min(self.as_vec(), other.as_vec()));
        }

        /// Clamps each element to [lower_bound, upper_bound]
        /// Note: This is not gauranteed to observe IEEE 754.
        /// on x86_64 it will probably emit `vmaxps`/`vminps` which do not.
        pub fn clamp(self: Self, lower_bound: FType, upper_bound: FType) Self {
            if (lower_bound > upper_bound) @panic("called clamp with lower_bound > upper_bound");

            var res = self.as_vec();
            res = @min(res, splat(upper_bound).as_vec());
            res = @max(res, splat(lower_bound).as_vec());
            return @bitCast(res);
        }

        /// Returns vector with the sign of each element, represented as -1.0 / 0 / 1.0
        pub fn signs(self: Self) Self {
            // todo: perf
            return init(std.math.sign(self.x), std.math.sign(self.y), std.math.sign(self.z));
        }

        /// keeps absolute values of each element of self, but signs them with the signs
        /// of the elements of other.
        /// Example:
        /// ``` zig
        /// const a = Vec3A.init(1, 2, 3);
        /// const b = Vec3A.init(-6, 5, -4);
        /// _ = a.copysign(b); // -> { -1, 2, -3 }
        /// ```
        pub fn copysign(self: Self, other: Self) Self {
            // todo: perf
            return self.abs().mul(other.signs());
        }

        /// Element-wise clamp from [0, 1]
        pub fn saturate(self: Self) Self {
            // todo: perf maybe
            return self.clamp(0, 1);
        }

        /// Square of the length of the vector, skips computing sqrt
        pub fn length_squared(self: Self) FType {
            return self.mul(self).sum();
        }

        /// Length of the vector
        pub fn length(self: Self) FType {
            return @sqrt(self.length_squared()); // i BELIEVE in llvm inlining
        }

        /// Reciprocal of the length of the vector (1 / length)
        pub fn length_recip(self: Self) FType {
            return ONE.div(self.length());
        }

        /// Sets length of vector to `upper_bound` if it is exceeding `upper_bound`, otherwise does nothing.
        /// Does not affect the direction of the vector. `upper_bound` must be zero or positive.
        pub fn clamp_length_max(self: Self, upper_bound: FType) Self {
            if (upper_bound < 0) @panic("clamp_length_max upper_bound must be >= 0");

            const sqr_length = self.length_squared();
            const sqr_upper_bound = upper_bound * upper_bound;

            if (sqr_length > sqr_upper_bound) {
                @setFloatMode(.optimized);
                return self.mul_scalar(upper_bound / @sqrt(sqr_length));
            } else {
                return self;
            }
        }

        /// Sets length of vector to `lower_bound` if it is less-than `lower_bound`, otherwise does nothing.
        /// Does not affect the direction of the vector. `lower_bound` must be zero or positive.
        pub fn clamp_length_min(self: Self, lower_bound: FType) Self {
            if (lower_bound < 0) @panic("clamp_length_min lower_bound must be >= 0");

            const sqr_length = self.length_squared();
            const sqr_lower_bound = lower_bound * lower_bound;

            if (sqr_length < sqr_lower_bound) {
                @setFloatMode(.optimized);
                return self.mul_scalar(lower_bound / @sqrt(sqr_length));
            } else {
                return self;
            }
        }

        /// Sets length of vector to `lower_bound` if it is less-than `lower_bound`, and sets
        /// length to `upper_bound` if its greater-than `upper_bound`. If it is already within
        /// this inclusive range this will have no effect.
        /// Does not affect the direction of the vector.
        /// `lower_bound` and `upper_bound` must be zero or positive.
        pub fn clamp_length(self: Self, lower_bound: FType, upper_bound: FType) Self {
            return self.clamp_length_max(upper_bound).clamp_length_min(lower_bound);
        }

        /// Scales vector so that length is `len`, does not affect direction. `len` must be >= 0;
        pub fn set_length(self: Self, len: FType) Self {
            if (len < 0) @panic("length cannot be < 0");
            @setFloatMode(.optimized);
            return self.mul_scalar(len / self.length_recip());
        }

        /// Distance from self -> other
        /// When called as a method you can read this as "distanceTo"
        pub fn distance_squared(self: Self, other: Self) FType {
            const diff = other.sub(self);
            return diff.mul(diff).sum();
        }

        /// Distance from self -> other
        /// When called as a method you can read this as "distanceTo"
        pub fn distance(self: Self, other: Self) FType {
            @setFloatMode(.optimized);
            return @sqrt(distance(self, other));
        }

        /// One over the distance of self -> other
        /// When called as a method you can read this as "distanceRecipTo"
        pub fn distance_recip(self: Self, other: Self) FType {
            // todo: perf
            return ONE.div(distance(self, other));
        }

        /// Scales vector such that it is length=1, does not affect direction.
        /// PANICS if length is zero!
        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len == 0) @panic("tried to normalize zero length vector");
            return self.div_scalar(len);
        }

        /// Scales vector such that it is length=1, does not affect direction.
        /// Also returns the length, which was computed as a side-effect.
        /// PANICS if length is zero!
        pub fn normalize_and_length(self: Self) struct { vec: Self, length: FType } {
            const len = self.length();
            if (len == 0) @panic("tried to normalize zero length vector");
            return .{ .vec = self.div_scalar(len), .length = len };
        }

        /// Scales vector such that it is length=1, does not affect direction.
        /// Returns ZERO vector if length is zero!
        pub fn normalize_or_zero(self: Self) Self {
            const len = self.length();
            if (len == 0) return self.ZERO;
            return self.div_scalar(len);
        }
        
        /// Reorders elements of the vector.
        /// Example:
        /// ```zig
        /// const a = Vec3A.init(1, 2, 3);
        /// _ = a.swizzle("zzy"); // -> { 3, 3, 2 }
        /// ```
        pub fn swizzle(self: Self, comptime mask: []const u8) Self {
            if (mask.len != 3) @compileError("swizzle mask must be length equal to dimensions (3)");

            comptime var order: [4]isize = undefined;
            inline for (mask, 0..) |char, i| {
                order[i] = switch(char) {
                    'x' => 0, 'y' => 1, 'z' => 2,
                    else => @compileError("invalid axis label"),
                };
            }
            order[3] = 0;

            return @bitCast(@shuffle(FType, self.as_vec(), undefined, order));
        }

        /// Takes the cross product of self and other
        pub fn cross(self: Self, other: Self) Self {
            return sub(
                mul(self.swizzle("yzx"), other.swizzle("zxy")),
                mul(self.swizzle("zxy"), other.swizzle("yzx"))
            );
        }

        /// Takes vector projection of `self` onto `other`
        /// PANICS if `other` length is zero!
        pub fn project(self: Self, other: Self) Self {
            const other_length_squared = other.length_squared();
            if (other_length_squared == 0) @panic("tried to project onto zero length vector");
            return other.mul_scalar(self.dot(other) / other_length_squared);
        }

        /// Takes vector projection of `self` onto `other`
        /// returns ZERO vector if `other` length is zero!
        pub fn project_or_zero(self: Self, other: Self) Self {
            const other_length_squared = other.length_squared();
            if (other_length_squared == 0) return ZERO;
            return other.mul_scalar(self.dot(other) / other_length_squared);
        }

        /// Linearly interpolates between `self` and `other`
        /// At s=0 `self` will be returned.
        /// At s=1 `other` will be returned
        /// At s=0.5 should give same result as `self.midpoint(other)`
        /// At values (< 0 || > 1) we will further interpolate in the respective direction.
        pub fn lerp(self: Self, other: Self, s: FType) Self {
            const delta = other.sub(self);
            return self.add(delta.mul_scalar(s));
        }

        /// Computes midpoint of `self` and `other`.
        /// Should give same result as `self.lerp(other, 0.5)`
        pub fn midpoint(self: Self, other: Self) Self {
            return add(self, other).div_scalar(2);
        }

        /// Constructs a Vec2 of the same float-type by truncating - discarding z
        pub fn to_vec2_truncate(self: Self) vec2.Vector2(FType) {
            return @bitCast(
                @shuffle(FType, self.as_vec(), undefined, @Vector(2, FType){0, 1})
            );
        }

        /// Constructs a Vec4 of the same float-type with zero as the new w component - { x, y, z, 0 }
        pub fn to_vec4_zero_extend(self: Self) vec4.Vector4(FType) {
            std.debug.assert(self._pad == 0);
            return @bitCast(
                @shuffle(FType, self.as_vec(), undefined, @Vector(4, FType){0, 1, 2, 3})
            );
        }

        /// Creates newly sized vector (either 2, 3 or 4) out of arbitrary order
        /// Examples:
        /// ```zig
        /// const original = Vec3A.init(1, 2, 3);
        /// _ = original.swizzle_and_resize("yzxz") // -> Vec4  { 2, 3, 1, 3 }
        /// _ = original.swizzle_and_resize("x00z") // -> Vec4  { 1, 0, 0, 3 }
        /// _ = original.swizzle_and_resize("xx")   // -> Vec2  { 1, 1 }
        /// _ = original.swizzle_and_resize("zyx")  // -> Vec3A { 3, 2, 1 }
        /// ```
        pub fn swizzle_and_resize(self: Self, comptime mask: []const u8) 
            switch (mask.len) {
                2 => vec2.Vector2(FType),
                3 => Self,
                4 => vec4.Vector4(FType),
                else => @compileError("`swizzle_and_resize` mask length must be 2, 3 or 4"),
            }
        {
            const order_len = if (mask.len == 2) 2 else 4; // keep pad if making a vec3
            std.debug.assert(self._pad == 0);

            comptime var order: [order_len]isize = undefined;
            inline for (mask, 0..) |char, i| {
                order[i] = switch(char) {
                    'x' => 0, 'y' => 1, 'z' => 2, '0' => 3,
                    else => @compileError("invalid axis label (must be x, y, z or 0)"),
                };
            }
            if (mask.len == 3) order[3] = 0;

            return @bitCast(@shuffle(FType, self.as_vec(), undefined, order));
        }

        // todo: extends etc once more vectors are implemented
    };
}


test {
    const Vec3A  = Vector3A(f32);

    const t = std.testing;

    const base = Vec3A.init(1, 2, 3);
    const swoz = base.swizzle_and_resize("x0y");
    const should_be = Vec3A.init(1, 0, 2);
    try t.expect(swoz.eq(should_be));

    var asd = Vec3A.Y.neg().mul_scalar(100);
    _ = asd.swizzle("zyx").cross(Vec3A.X).normalize();
    asd = asd.swizzle("zyx").add(Vec3A.X).add(Vec3A.ONE);
    _ = asd.to_vec4_zero_extend();
    _ = asd.to_vec2_truncate();
    try t.expectEqual(2, asd.x);
    try t.expectEqual(-99, asd.y);
    try t.expectEqual(1, asd.z);

    var a = Vec3A.init(0, 1, 2).clamp(0.5, 1.5);
    try t.expectEqual(0.5, a.x);
    try t.expectEqual(1.0, a.y);
    try t.expectEqual(1.5, a.z);
    a = a.neg().abs();
    try t.expectEqual(0.5, a.x);
    try t.expectEqual(1.0, a.y);
    try t.expectEqual(1.5, a.z);

    var zz3 = a.swizzle_and_resize("xyz");
    const zz2 = zz3.swizzle_and_resize("xy");
    const zz4 = zz3.swizzle_and_resize("xyxy");
    _ = zz2;
    _ = zz4;

    _ = zz3.recip_sqrt_fast();

    const b = Vec3A.init(2, 0, 0);
    try t.expectEqual(1, b.clamp_length_max(1).x);
    try t.expectEqual(3, b.clamp_length_min(3).x);
    try t.expectEqual(2, b.clamp_length(2, 2).x);

    try t.expectEqual(2,b.normalize_and_length().length);

    _ = a.max(asd);
}

