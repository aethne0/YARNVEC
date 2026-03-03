const std = @import("std");

pub fn Vector3A(comptime FType: type) type {
    if (FType != f32 and FType != f64 and FType != f128) 
        @compileError("fType must be f32, f64, f128");

    return extern struct {
        const Self = @This();

        x:      FType align(@sizeOf(FType) * 4),
        y:      FType,
        z:      FType,
        _pad:   FType = 0,

        /// Initialize struct with values
        /// Example: `const some_vec = Vec3A.init(.{1, 2, 3});`
        pub fn init(x: FType, y: FType, z: FType) Self {
            return .{ .x = x, .y = y, .z = z };
        }

        pub inline fn splat(scalar: FType) Self {
            const result: @Vector(4, FType) = @splat(scalar);
            return @bitCast(result);
        }

        pub const ZERO      = splat(0);
        pub const ONE       = splat(1);
        pub const NEG_ONE   = splat(-1);
        pub const X         = init(1, 0, 0);
        pub const Y         = init(0, 1, 0);
        pub const Z         = init(0, 0, 1);
        pub const NEG_X     = init(-1, 0, 0);
        pub const NEG_Y     = init(0, -1, 0);
        pub const NEG_Z     = init(0, 0, -1);
        pub const MIN       = splat(std.math.floatMin(FType));
        pub const MAX       = splat(std.math.floatMax(FType));
        pub const NAN       = splat(std.math.nan(FType));
        pub const INF       = splat(std.math.inf(FType));
        pub const NEG_INF   = splat(-std.math.inf(FType));
        pub const AXES      = [_]Self{ X, Y, Z };

        inline fn as_vec(self: Self) @Vector(4, FType) {
            return @bitCast(self);
        }

        // This is to safely handle the pad element in some reduce operations (product, min, max)
        inline fn as_vec_3(self: Self) @Vector(3, FType) {
            return @shuffle(FType, self.as_vec(), undefined, @Vector(3, FType){0, 1, 2});
        }

        pub fn add(self: Self, other: Self) Self {
            return @bitCast( self.as_vec() + other.as_vec() );
        }

        pub fn sub(self: Self, other: Self) Self {
            return @bitCast( self.as_vec() - other.as_vec() );
        }

        pub fn mul(self: Self, other: Self) Self {
            return @bitCast(self.as_vec() * other.as_vec());
        }

        pub fn div(self: Self, other: Self) Self {
            var result: Self = @bitCast(self.as_vec() / other.as_vec());
            result._pad = 0;
            return result;
        }

        pub fn add_scalar(self: Self, scalar: FType) Self {
            return self.add(splat(scalar));
        }

        pub fn sub_scalar(self: Self, scalar: FType) Self {
            return self.sub(splat(scalar));
        }

        pub fn mul_scalar(self: Self, scalar: FType) Self {
            return self.mul(splat(scalar));
        }

        pub fn div_scalar(self: Self, scalar: FType) Self {
            return self.div(splat(scalar));
        }

        pub fn neg(self: Self) Self {
            return self.mul_scalar(-1);
        }

        pub fn sum(self: Self) FType {
            std.debug.assert(self._pad == 0);
            return @reduce(.Add, self.as_vec());
        }

        pub fn product(self: Self) FType {
            return @reduce(.Mul, self.as_vec_3());
        }

        pub fn min_element(self: Self) FType {
            return @reduce(.Min, self.as_vec_3());
        }

        pub fn max_element(self: Self) FType {
            return @reduce(.Max, self.as_vec_3());
        }

        pub fn abs(self: Self) Self {
            return @bitCast(@abs(self.as_vec()));
        }

        pub fn dot(self: Self, other: Self) FType {
            std.debug.assert(self._pad == 0);
            std.debug.assert(other._pad == 0);
            return self.mul(other).sum();
        }

        pub fn max(self: Self, other: Self) Self {
            return @bitCast(@max(self.as_vec(), other.as_vec()));
        }

        pub fn min(self: Self, other: Self) Self {
            return @bitCast(@min(self.as_vec(), other.as_vec()));
        }

        /// Note: This is not gauranteed to observe IEEE 754.
        /// on x86_64 it will probably emit `vmaxps`/`vminps` which do not.
        pub fn clamp(self: Self, lower_bound: FType, upper_bound: FType) Self {
            if (lower_bound > upper_bound) @panic("called clamp with lower_bound > upper_bound");

            var res = self.as_vec();
            res = @min(res, splat(upper_bound).as_vec());
            res = @max(res, splat(lower_bound).as_vec());
            return @bitCast(res);
        }

        pub fn length_squared(self: Self) FType {
            return self.mul(self).sum();
        }

        pub fn length(self: Self) FType {
            return @sqrt(self.length_squared()); // i BELIEVE in llvm inlining
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
            return @sqrt(distance(self, other));
        }

        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len == 0) @panic("tried to normalize zero length vector");
            return self.div_scalar(len);
        }

        pub fn normalize_or_zero(self: Self) Self {
            const len = self.length();
            if (len == 0) return self.ZERO;
            return self.div_scalar(len);
        }
        
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

        pub fn cross(self: Self, other: Self) Self {
            return sub(
                mul(self.swizzle("yzx"), other.swizzle("zxy")),
                mul(self.swizzle("zxy"), other.swizzle("yzx"))
            );
        }

        pub fn project(self: Self, other: Self) Self {
            const other_length_squared = other.length_squared();
            if (other_length_squared == 0) @panic("tried to project onto zero length vector");
            return other.mul_scalar(self.dot(other) / other_length_squared);
        }

        pub fn project_or_zero(self: Self, other: Self) Self {
            const other_length_squared = other.length_squared();
            if (other_length_squared == 0) return ZERO;
            return other.mul_scalar(self.dot(other) / other_length_squared);
        }

        // todo: extends etc once more vectors are implemented
    };
}


test {
    const Vec3A  = Vector3A(f32);

    const t = std.testing;

    _ = Vec3A.init(1, 2, 3);
    var asd = Vec3A.Y.neg().mul_scalar(100);
    _ = asd.swizzle("zyx").cross(Vec3A.X).normalize();
    asd = asd.swizzle("zyx").add(Vec3A.X).add(Vec3A.ONE);
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

    _ = a.max(asd);
}

