const std = @import("std");
const vector_common = @import("vector_common.zig");

pub fn Vector3A(comptime FType: type) type {
    if (FType != f32 and FType != f64) @compileError("fType must be f32 | f64");

    const BitsType = if (FType == f32) u96 else u192;
    const V2Type = @import("vector_2.zig").Vector2(FType);
    const V4Type = @import("vector_4.zig").Vector4(FType);

    const VRawType = @Vector(4, FType);

    return extern struct {
        const Self = @This();
        const shared = vector_common.VectorAlignedCommon(FType, 3, Self);

        x:      FType align(@sizeOf(FType) * 4),
        y:      FType,
        z:      FType,
        _pad:   FType = 0,

        /// Initialize vector with value for each dimension
        /// Example: `const some_vec = Vec3A.init(1, 2, 3);`
        pub fn init(x: FType, y: FType, z: FType) Self {
            return .{ .x = x, .y = y, .z = z };
        }

        const as_vec = shared.as_vec;

        pub const splat = shared.splat;

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
        /// array of all positive unit vectors
        pub const AXES      = [_]Self{ X, Y, Z };


        pub const ZERO = shared.ZERO;
        pub const ONE = shared.ONE;
        pub const NEG_ONE = shared.NEG_ONE;
        pub const MIN = shared.MIN;
        pub const MAX = shared.MAX;
        pub const NAN = shared.NAN;
        pub const INF = shared.INF;
        pub const NEG_INF = shared.NEG_INF;

        pub const add = shared.add;
        pub const sub = shared.sub;
        pub const mul = shared.mul;
        pub const div = shared.div;
        pub const mulAdd = shared.mulAdd;
        pub const addScalar  = shared.addScalar;
        pub const subScalar  = shared.subScalar;
        pub const mulByScalar  = shared.mulByScalar;
        pub const divByScalar  = shared.divByScalar;
        pub const neg  = shared.neg;
        pub const ceil  = shared.ceil;
        pub const round  = shared.round;
        pub const floor  = shared.floor;
        pub const sin  = shared.sin;
        pub const cos  = shared.cos;
        pub const ln  = shared.ln;
        pub const log2  = shared.log2;
        pub const exp  = shared.exp;
        pub const exp2  = shared.exp2;
        pub const recip  = shared.recip;
        pub const sqrt  = shared.sqrt;
        pub const recipSqrtFast  = shared.recipSqrtFast;
        pub const abs  = shared.abs;
        pub const min  = shared.min;
        pub const max  = shared.max;
        pub const saturate = shared.saturate;
        pub const lengthSquared = shared.lengthSquared;
        pub const length = shared.length;
        pub const lengthRecip = shared.lengthRecip;
        pub const clampLengthMax = shared.clampLengthMax;
        pub const clampLengthMin = shared.clampLengthMin;
        pub const clampLength = shared.clampLength;
        pub const setLength = shared.setLength;
        pub const distanceSquared = shared.distanceSquared;
        pub const distance = shared.distance;
        pub const distanceReciprocal = shared.distanceReciprocal;
        pub const normalize = shared.normalize;
        pub const normalizeAndLength = shared.normalizeAndLength;
        pub const normalizeOrZero = shared.normalizeOrZero;
        pub const isNormalized = shared.isNormalized;
        pub const lerp = shared.lerp;
        pub const midpoint = shared.midpoint;
        pub const swizzle = shared.swizzle;
        pub const swizzleResize = shared.swizzleResize;

        // This is to safely handle the pad element in some reduce operations (product, min, max)
        inline fn as_vec_3(self: Self) @Vector(3, FType) {
            return @shuffle(FType, self.as_vec(), undefined, @Vector(3, FType){0, 1, 2});
        }

        /// Element-wise bitwise equality
        pub fn eq(self: Self, other: Self) bool {
            // reference:
            // const result = erch.x86._mm_add_ps(self.as_vec(), other.as_vec());
            // return 0 != arch.x86._mm_testz_ps(result, result);

            // todo: perf
            const left = @as(BitsType, @bitCast(self.as_vec_3()));
            const right = @as(BitsType, @bitCast(other.as_vec_3()));
            return left == right;
        }

        /// Computes dot product with another vector
        pub fn dot(self: Self, other: Self) FType {
            return self.mul(other).sum();
        }

        /// Sum of all elements
        pub fn sum(self: Self) FType {
            return @reduce(.Add, self.as_vec_3());
        }

        /// Clamps each element to [lower_bound, upper_bound]
        /// Note: This is not gauranteed to observe IEEE 754.
        /// on x86_64 it will probably emit `vmaxps`/`vminps` which do not.
        pub fn clampByScalars(self: Self, lower_bound: FType, upper_bound: FType) Self {
            if (lower_bound > upper_bound)
                std.debug.panic(
                    \\ called clamp with lower_bound > upper_bound:
                    \\ lower: {any}, upper: {any}
                    , .{lower_bound, upper_bound});

            var res = self.as_vec();
            res = @min(res, splat(upper_bound).as_vec());
            res = @max(res, splat(lower_bound).as_vec());
            res[3] = 0;
            return @bitCast(res);
        }

        /// Element-wise clamp
        /// Note: This is not gauranteed to observe IEEE 754.
        /// on x86_64 it will probably emit `vmaxps`/`vminps` which do not.
        pub fn clamp(self: Self, lower_bound: Self, upper_bound: Self) Self {
            // todo: perf
            if (!@reduce(.And, upper_bound.as_vec_3() >= lower_bound.as_vec_3()))
                std.debug.panic(
                    \\ called clamp with lower_bound > upper_bound (for one or more elements):
                    \\ lower: {any}, upper: {any}
                    , .{lower_bound, upper_bound});
            
            return self.min(upper_bound).max(lower_bound);
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
            if (other_length_squared == 0) std.debug.panic("tried to project onto zero length vector", .{});
            return other.mul_scalar(self.dot(other) / other_length_squared);
        }

        // TODO: doc
        /// Takes vector projection of `self` onto `other`
        /// returns ZERO vector if `other` length is zero!
        pub fn projectOrZero(self: Self, other: Self) Self {
            const other_length_squared = other.length_squared();
            if (other_length_squared == 0) return ZERO;
            return other.mul_scalar(self.dot(other) / other_length_squared);
        }

        pub fn reflect(self: Self, normal: Self) Self {
            if (!normal.is_normalized()) std.debug.panic("normal must be normalized (length=1)\n", .{});
            return self.sub(normal.mul_scalar(2 * self.dot(normal)));
        }

        // TODO: doc
        // PANICS if self isnt normalized
        // PANICS if normal isnt normalized
        pub fn refract(self: Self, normal: Self, eta: FType) Self {
            if (!self.is_normalized()) std.debug.panic("self must be normalized (length=1)\n", .{});
            if (!normal.is_normalized()) std.debug.panic("normal must be normalized (length=1)\n", .{});

            // TODO: check this cause i dont even know what it does
            const n_dot_i = normal.dot(self);
            const k = 1 - eta * eta * (1 - n_dot_i * n_dot_i);
            if (k >= 0) {
                return self.mul_scalar(eta).sub(normal.mul_scalar(eta * n_dot_i + @sqrt(k)));
            } else {
                return ZERO;
            }
        }

        // TODO: doc
        pub fn angleBetween(self: Self, other: Self) FType {
            // TODO: make approx acos
            std.math.acos(
                self.dot(other) / @sqrt(self.length_squared() * other.length_squared())
            );
        }

        /// Rotates self about the X axis by `angle` (radians)
        pub fn rotateX(self: Self, angle: FType) Self {
            // todo: perf
            const sin_angle = @sin(angle);
            const cos_angle = @cos(angle);

            return init(
                self.x,
                self.y * cos_angle - self.z * sin_angle,
                self.y * sin_angle + self.z * cos_angle,
            );
        }

        /// Rotates self about the Y axis by `angle` (radians)
        pub fn rotateY(self: Self, angle: FType) Self {
            // todo: perf
            const sin_angle = @sin(angle);
            const cos_angle = @cos(angle);

            return init(
                self.x * cos_angle + self.z * sin_angle,
                self.y,
                self.z * cos_angle - self.x * sin_angle 
            );
        }

        /// Rotates self about the Z axis by `angle` (radians)
        pub fn rotateZ(self: Self, angle: FType) Self {
            // todo: perf
            const sin_angle = @sin(angle);
            const cos_angle = @cos(angle);

            return init(
                self.x * cos_angle - self.y * sin_angle,
                self.x * sin_angle - self.y * cos_angle ,
                self.z,
            );
        }

        /// Product of all elements
        pub fn product(self: Self) FType {
            // todo: perf
            // glam uses some slick shuffling to do this but this seems to emit better for x86_64 and aarch64
            // https://docs.rs/glam/0.32.0/src/glam/f32/sse2/vec3a.rs.html#418
            // https://godbolt.org/z/W4v8oYWx9
            // When we try to recreate this through @builtins we get quite a suboptimal result
            // https://godbolt.org/z/Wh8W6EEn1
            //
            // The below method of "casting out" the pad element gives a pretty good result, better
            // than trying to use @builtins, but ill have to benchmark it against the asm of the glam
            // version. 
            //
            // This will be a similar case for a lot of these operations where the pad element would
            // screw up the result (min/max/product).
            return @reduce(.Mul, self.as_vec_3());
        }

        /// min of all elements
        pub fn minElement(self: Self) FType {
            return @reduce(.Min, self.as_vec_3());
        }

        /// max of all elements
        pub fn maxElement(self: Self) FType {
            return @reduce(.Max, self.as_vec_3());
        }

        /// Constructs a Vec2 of the same float-type by truncating - discarding z
        pub fn toVec2Truncate(self: Self) V2Type {
            return @bitCast(
                @shuffle(FType, self.as_vec(), undefined, @Vector(2, FType){0, 1})
            );
        }

        /// Constructs a Vec4 of the same float-type with zero as the new w component - { x, y, z, 0 }
        pub fn toVec4ZeroExtend(self: Self) V4Type {
            // todo: perf
            var result: VRawType = @shuffle(FType, self.as_vec(), undefined, VRawType{0, 1, 2, 3});
            result[3] = 0;
            return @bitCast(result);
        }

        // todo: extends etc once more vectors are implemented
    };
}

/// aligned 3-dimensional f32 vector
pub const Vec3A = Vector3A(f32);
/// aligned 3-dimensional f64 vector
pub const Vec3Af64 = Vector3A(f64);


// ▄▄▄▄▄▄▄▄ ..▄▄ · ▄▄▄▄▄.▄▄ · 
// •██  ▀▄.▀·▐█ ▀. •██  ▐█ ▀. 
//  ▐█.▪▐▀▀▪▄▄▀▀▀█▄ ▐█.▪▄▀▀▀█▄
//  ▐█▌·▐█▄▄▌▐█▄▪▐█ ▐█▌·▐█▄▪▐█
//  ▀▀▀  ▀▀▀  ▀▀▀▀  ▀▀▀  ▀▀▀▀ 

fn Vec3ATests (FType: type) type {
    const t = std.testing;

    return struct  {
        const VType = Vector3A(FType);

        fn expectEqualVec(expected: VType, actual: VType) !void {
            return t.expect(VType.eq(expected, actual));
        }

        test "clamp" {
            var a = VType.init(0, 1, 2).clamp(VType.ZERO, VType.ONE);
            try t.expectEqual(0.0, a.x);
            try t.expectEqual(1.0, a.y);
            try t.expectEqual(1.0, a.z);
            try t.expectEqual(0, a._pad);
        }

        test "div_by_zero" {
            _ = VType.ONE.div(VType.ZERO);
        }

        test "clamp_by_scalars" {
            try expectEqualVec(
                VType.init(0.5, 1, 1.5),
                VType.init(0, 1, 2).clampByScalars(0.5, 1.5)
            );
        }

        test "swiz_resize" {
            const swozzled = VType.init(1, 2, 3).swizzleResize("xxy");
            try expectEqualVec(swozzled, VType.init(1, 1, 2));
        }

        test "chungus" {
            var asd = VType.Y.neg().mulByScalar(100);
            _ = asd.swizzle("zyx").cross(VType.X).normalize();
            asd = asd.swizzle("zyx").add(VType.X).add(VType.ONE);
            _ = asd.toVec4ZeroExtend();
            _ = asd.toVec2Truncate();
            try t.expectEqual(2, asd.x);
            try t.expectEqual(-99, asd.y);
            try t.expectEqual(1, asd.z);

            var a = VType.init(0, 1, 2).clampByScalars(0.5, 1.5);
            try t.expectEqual(0.5, a.x);
            try t.expectEqual(1.0, a.y);
            try t.expectEqual(1.5, a.z);
            a = a.neg().abs();
            try t.expectEqual(0.5, a.x);
            try t.expectEqual(1.0, a.y);
            try t.expectEqual(1.5, a.z);

            var zz3 = a.swizzleResize("xyz");
            try t.expectEqual(0.5, zz3.x);
            try t.expectEqual(1.0, zz3.y);
            try t.expectEqual(1.5, zz3.z);

            _ = zz3.recipSqrtFast();

            const zz2 = zz3.swizzleResize("xy");
            const zz4 = zz3.swizzleResize("xyxy");
            _ = zz2;
            _ = zz4;

            const b = VType.init(2, 0, 0);
            try t.expectEqual(1, b.clampLengthMax(1).x);
            try t.expectEqual(3, b.clampLengthMin(3).x);
            try t.expectEqual(2, b.clampLength(2, 2).x);

            try t.expectEqual(2,b.normalizeAndLength().length);

            _ = a.max(asd);
        }
    };
}

comptime {
    _ = Vec3ATests(f32);
    _ = Vec3ATests(f64);
}

