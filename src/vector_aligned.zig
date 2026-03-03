const std = @import("std");

fn VectorAligned(comptime CT_ftype: type, comptime CT_dims: usize) type {
    if (CT_dims != 3 and CT_dims != 4) 
        @compileError("VectorAligned only for dim 3,4");

    if (CT_ftype != f32 and CT_ftype != f64 and CT_ftype != f128) 
        @compileError("fType must be f32, f64, f128");

    const CT_size = @sizeOf(CT_ftype) * 4;

    const has_pad   = CT_dims == 3;
    const has_w     = CT_dims == 4;

    return extern struct {
        const Self = @This();

        // 3: (x, y, z, _pad)   4 * fType
        // 4: (x, y, z, w)      4 * fType
        x:      CT_ftype align(CT_size),
        y:      CT_ftype,
        z:      CT_ftype,
        _pad:   if (has_pad)    CT_ftype else void = if (has_pad) 0 else {},
        w:      if (has_w)      CT_ftype else void = if (has_w)   0 else {}, 


        pub const init = switch (CT_dims) {
            2 => struct { 
                fn init(x: CT_ftype, y: CT_ftype) Self {
                    return .{ .x = x, .y = y };
                }
            }.init,
            3 => struct { 
                fn init(x: CT_ftype, y: CT_ftype, z: CT_ftype) Self {
                    return .{ .x = x, .y = y, .z = z };
                }
            }.init,
            4 => struct { 
                fn init(x: CT_ftype, y: CT_ftype, z: CT_ftype, w: CT_ftype) Self {
                    return .{ .x = x, .y = y, .z = z, .w = w };
                }
            }.init,
            else => unreachable,
        };

        pub fn splat(scalar: CT_ftype) Self {
            const result: @Vector(4, CT_ftype) = @splat(scalar);
            return @bitCast(result);
        }

        pub const ZERO = splat(0);
        pub const ONE = splat(1);
        pub const NEG_ONE = splat(-1);


        inline fn as_vec_ptr(self: anytype)
            if (@typeInfo(@TypeOf(self)).pointer.is_const) 
                *const @Vector(4, CT_ftype) 
            else 
                *@Vector(4, CT_ftype)
        {
            return @ptrCast(self);
        }

        inline fn as_vec_ptr_unaligned(self: anytype)
            if (@typeInfo(@TypeOf(self)).pointer.is_const) 
                *const @Vector(CT_dims, CT_ftype) 
            else 
                *@Vector(CT_dims, CT_ftype)
        {
            return @ptrCast(self);
        }


        pub fn axis_unit(comptime axis_label: []const u8) Self {
            if (axis_label.len != 1) @compileError("Must be label of dimension axis, like \"x\"");

            const axis_index = switch (axis_label[0]) {
                'x' => 0,
                'y' => 1,
                'z' => 2,
                'w' => 3,
                else => @compileError("invalid axis label"),
            };

            if (axis_index + 1 > CT_dims) @compileError("passed axis label is of a greater dimension than vector");

            var result: @Vector(4, CT_ftype) = @splat(0);
            result[axis_index] = 1;
            return @bitCast(result);
        }

        pub fn add(self: Self, other: Self) Self {
            return @bitCast( self.as_vec_ptr().* + other.as_vec_ptr().* );
        }

        pub fn sub(self: Self, other: Self) Self {
            return @bitCast( self.as_vec_ptr().* - other.as_vec_ptr().* );
        }

        pub fn mul(self: Self, other: Self) Self {
            return @bitCast( self.as_vec_ptr().* * other.as_vec_ptr().* );
        }

        pub fn div(self: Self, other: Self) Self {
            if (CT_dims == 3) {
                var result: Self = @bitCast(self.as_vec_ptr().* / other.as_vec_ptr().*);
                result._pad = 0;
                return result;
            } else {
                return @bitCast( self.as_vec_ptr().* / other.as_vec_ptr().* );
            }
        }

        pub fn add_scalar(self: Self, scalar: CT_ftype) Self {
            return self.add(splat(scalar));
        }

        pub fn sub_scalar(self: Self, scalar: CT_ftype) Self {
            return self.sub(splat(scalar));
        }

        pub fn mul_scalar(self: Self, scalar: CT_ftype) Self {
            return self.mul(splat(scalar));
        }

        pub fn div_scalar(self: Self, scalar: CT_ftype) Self {
            return self.div(splat(scalar));
        }

        pub fn neg(self: Self) Self {
            return self.mul_scalar(-1);
        }

        pub fn sum(self: Self) CT_ftype {
            return @reduce(.Add, self.as_vec_ptr_unaligned().*);
        }

        pub fn product(self: Self) CT_ftype {
            return @reduce(.Mul, self.as_vec_ptr_unaligned().*);
        }

        pub fn max(self: Self) CT_ftype {
            return @reduce(.Max, self.as_vec_ptr_unaligned().*);
        }

        pub fn min(self: Self) CT_ftype {
            return @reduce(.Min, self.as_vec_ptr_unaligned().*);
        }

        pub fn dot(self: Self, other: Self) CT_ftype {
            return self.mul(other).sum();
        }

        pub fn length_squared(self: Self) CT_ftype {
            return self.mul(self).sum();
        }

        pub fn length(self: Self) CT_ftype {
            return @sqrt(self.length_squared()); // i BELIEVE in llvm inlining
        }

        /// Distance from self -> other
        /// When called as a method you can read this as "distanceTo"
        pub fn distance_squared(self: Self, other: Self) CT_ftype {
            const diff = other.sub(self);
            return diff.mul(diff).sum();
        }

        /// Distance from self -> other
        /// When called as a method you can read this as "distanceTo"
        pub fn distance(self: Self, other: Self) CT_ftype {
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
            if (mask.len != CT_dims) @compileError("swizzle mask must be length equal to dimensions");

            comptime var order: [4]isize = undefined;
            inline for (mask, 0..) |char, i| {
                order[i] = switch(char) {
                    'x' => 0, 'y' => 1, 'z' => 2, 'w' => 3,
                    else => @compileError("invalid axis label"),
                };
                if (order[i] + 1 > CT_dims) @compileError("passed axis label is of a greater dimension than vector");
            }

            if (CT_dims == 3) order[3] = 0;

            return @bitCast(@shuffle(CT_ftype, self.as_vec_ptr().*, undefined, order));
        }

        pub fn cross(self: Self, other: Self) Self {
            if (CT_dims != 3) 
                @compileError(
                    \\ `cross` product method only is available for dimension-3 vectors. 
                    \\ This method is defined on dimension-4 vectors because conditionally 
                    \\ including it at comptime seemed to degrade ZLS tooling after calling it. 
                    \\ Possibly cause the only way is to declare it with `pub const` as opposed
                    \\ to `pub fn` like normal methods
                    \\
                    \\ We apologize for any inconvenience!
                    \\ -yarn & the yarnvec team
                );
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
    };
}

/// aligned 3-dimensional f32 vector
pub const Vec3A  = VectorAligned(f32, 3);
/// 4-dimensional f32 vector
pub const Vec4  = VectorAligned(f32, 4);
/// aligned 3-dimensional f64 vector
pub const Vec3Af64  = VectorAligned(f64, 3);
/// 4-dimensional f64 vector
pub const Vec4f64  = VectorAligned(f64, 4);
/// aligned 3-dimensional f128 vector
pub const Vec3Af128 = VectorAligned(f128, 3);
/// 4-dimensional f128 vector
pub const Vec4f128 = VectorAligned(f128, 4);

test {
    _ = Vec3A;  _ = Vec4; 
    _ = Vec3Af64;  _ = Vec4f64; 
    _ = Vec3Af128; _ = Vec4f128;

    const t = std.testing;

    {
        var asd = Vec3A.axis_unit("y").neg().mul_scalar(100);
        _ = asd.swizzle("zyx").cross(Vec3A.axis_unit("x")).normalize();
        asd = asd.swizzle("zyx").add(Vec3A.axis_unit("x")).add(Vec3A.ONE);
        try t.expectEqual(2, asd.x);
        try t.expectEqual(-99, asd.y);
        try t.expectEqual(1, asd.z);
    }

    {
        var asd = Vec4.axis_unit("z").neg().mul_scalar(100);
        _ = asd.swizzle("wxzy").mul(asd).project(asd);
    }

    if (@sizeOf(Vec4f128) != 64) @compileError("huh");
}

