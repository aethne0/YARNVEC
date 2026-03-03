const std = @import("std");

pub fn Vector2(comptime FType: type) type {
    if (FType != f32 and FType != f64 and FType != f128) 
        @compileError("fType must be f32, f64, f128");

    return extern struct {
        const Self = @This();

        x:      FType align(@sizeOf(FType) * 2),
        y:      FType,

        /// Initialize struct with values
        /// Example: `const some_vec = Vec3A.init(.{1, 2, 3});`
        pub fn init(x: FType, y: FType) Self {
            return .{ .x = x, .y = y };
        }

        pub inline fn splat(scalar: FType) Self {
            const result: @Vector(2, FType) = @splat(scalar);
            return @bitCast(result);
        }

        pub const ZERO      = splat(0);
        pub const ONE       = splat(1);
        pub const NEG_ONE   = splat(-1);
        pub const X         = init(1, 0);
        pub const Y         = init(0, 1);
        pub const NEG_X     = init(-1, 0);
        pub const NEG_Y     = init(0, -1);
        pub const MIN       = splat(std.math.floatMin(FType));
        pub const MAX       = splat(std.math.floatMax(FType));
        pub const NAN       = splat(std.math.nan(FType));
        pub const INF       = splat(std.math.inf(FType));
        pub const NEG_INF   = splat(-std.math.inf(FType));
        pub const AXES      = [_]Self{ X, Y };

        inline fn as_vec(self: Self) @Vector(2, FType) {
            return @bitCast(self);
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
            return @bitCast(self.as_vec() / other.as_vec());
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
            if (mask.len != 2) @compileError("swizzle mask must be length equal to dimensions (2)");

            comptime var order: [2]isize = undefined;
            inline for (mask, 0..) |char, i| {
                order[i] = switch(char) {
                    'x' => 0, 'y' => 1,
                    else => @compileError("invalid axis label"),
                };
            }

            return @bitCast(@shuffle(FType, self.as_vec(), undefined, order));
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

        // todo: perp_dot
    };
}


test {
    const Vec2  = Vector2(f32);

    const t = std.testing;

    const some = Vec2.init(1, 2);
    try t.expectEqual(1, some.x);
    try t.expectEqual(2, some.y);

    // todo tests
}

