//!  ▄· ▄▌ ▄▄▄· ▄▄▄   ▐ ▄  ▌ ▐·▄▄▄ . ▄▄· 
//! ▐█▪██▌▐█ ▀█ ▀▄ █·•█▌▐█▪█·█▌▀▄.▀·▐█ ▌▪
//! ▐█▌▐█▪▄█▀▀█ ▐▀▀▄ ▐█▐▐▌▐█▐█•▐▀▀▪▄██ ▄▄
//!  ▐█▀·.▐█ ▪▐▌▐█•█▌██▐█▌ ███ ▐█▄▄▌▐███▌
//!   ▀ •  ▀  ▀ .▀  ▀▀▀ █▪. ▀   ▀▀▀ ·▀▀▀ 
const std = @import("std");

/// Row-major
pub const Matrix44 = extern struct {
    const Self = @This();

    // Layout wise this is a [4]@Vector(4, f32)
    v00: f32 align(64), v10: f32, v20: f32, v30: f32,
    v01: f32,           v11: f32, v21: f32, v31: f32,
    v02: f32,           v12: f32, v22: f32, v32: f32,
    v03: f32,           v13: f32, v23: f32, v33: f32,

    pub fn init(vals: [16]f32) Self {
        var mat: Self = undefined;
        inline for (0..16) |i| {
            mat.v_16()[i] = vals[i];
        }
        return mat;
    }

    pub fn init_aligned(vals: *align(16) [16]f32) Self {
        return @bitCast(vals);
    }

    pub fn splat(val: f32) Self {
        return init(@splat(val));
    }

    pub const IDENTITY = Self.init(
    [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    });

    pub const ZERO = Self.splat(0);
    pub const ONE = Self.splat(1);

    /// Cast to 4 packed 4-element f32 vectors
    inline fn v_4x4(self: anytype) 
    if (@typeInfo(@TypeOf(self)).pointer.is_const) 
        *const [4]@Vector(4, f32) 
    else 
        *[4]@Vector(4, f32)
    {
        return @ptrCast(self);
    }

    /// Cast to 16-element f32 vector
    inline fn v_16(self: anytype) 
    if (@typeInfo(@TypeOf(self)).pointer.is_const) 
        *const @Vector(16, f32) 
    else 
        *@Vector(16, f32)
    {
        return @ptrCast(self);
    }

    /// Element-wise
    pub fn add(self: Self, other: Self) Self {
        return @bitCast(self.v_16().* + other.v_16().*);
    }

    /// Element-wise
    pub fn sub(self: Self, other: Self) Self {
        return @bitCast(self.v_16().* - other.v_16().*);
    }

    /// Element-wise
    pub fn mul(self: Self, other: Self) Self {
        return @bitCast(self.v_16().* * other.v_16().*);
    }

    /// Element-wise
    pub fn div(self: Self, other: Self) Self {
        return @bitCast(self.v_16().* / other.v_16().*);
    }

    pub fn add_scalar(self: Self, scalar: f32) Self {
        return add(self, splat(scalar));
    }

    pub fn sub_scalar(self: Self, scalar: f32) Self {
        return sub(self, splat(scalar));
    }

    pub fn mul_scalar(self: Self, scalar: f32) Self {
        return mul(self, splat(scalar));
    }

    pub fn div_scalar(self: Self, scalar: f32) Self {
        return div(self, splat(scalar));
    }

    pub fn reduce_sum(self: Self) f32 {
        return @reduce(.Add, self.v_16().*);
    }

    pub fn transpose(self: Self) Self {
        return @bitCast(
            @shuffle(f32, self.v_16().*, undefined, 
                [_]i32{
                0,  4,  8, 12,
                1,  5,  9, 13,
                2,  6, 10, 14,
                3,  7, 11, 15,
            })
        );
    }

    pub fn init_from_row_major(vals: [16]f32) Self {
        return init(vals).transpose();
    }

    pub fn rotation_x(angle: f32) Self {
        var mat = IDENTITY;
        mat.v_16()[ 5] =  @cos(angle); mat.v_16()[ 6] = -@sin(angle);
        mat.v_16()[ 9] =  @sin(angle); mat.v_16()[10] =  @cos(angle);
        return mat;
    }

    pub fn rotation_y(angle: f32) Self {
        var mat = IDENTITY;
        mat.v_16()[ 0] =  @cos(angle); mat.v_16()[ 2] =  @sin(angle);
        mat.v_16()[ 8] = -@sin(angle); mat.v_16()[10] =  @cos(angle);
        return mat;
    }

    pub fn rotation_z(angle: f32) Self {
        var mat = IDENTITY;
        mat.v_16()[ 0] =  @cos(angle); mat.v_16()[ 1] = -@sin(angle);
        mat.v_16()[ 4] =  @sin(angle); mat.v_16()[ 5] =  @cos(angle);
        return mat;
    }

    pub fn matmul(self: Self, other: Self) Self {
        // ~ 17.25 cycles on ryzen 3800xt
        var result: Self = undefined;
        const a = self.v_4x4();
        const b = other.v_4x4();

        inline for (0..4) |i| {
            const col = b[i];
            
            var sum: @Vector(4, f32) = a[0] * @as(@Vector(4, f32), @splat(col[0]));
            sum = @mulAdd(@Vector(4, f32), a[1], @splat(col[1]), sum);
            sum = @mulAdd(@Vector(4, f32), a[2], @splat(col[2]), sum);
            sum = @mulAdd(@Vector(4, f32), a[3], @splat(col[3]), sum);
            
            result.v_4x4()[i] = sum;
        }
        
        return result;
    }

    /// temp debug
    pub fn print(self: Self) void {
        inline for (0..4) |i| {
            const row = self.v_4x4()[i];
            std.debug.print("{} {} {} {}\n", .{row[0], row[1], row[2], row[3]});
        }
    }
};

/// 16-byte aligned 3-dim f32 Vector
pub const Vector3 = extern struct {
    const Self = @This();

    x: f32 align(16), // Layout wise this is a @Vector(4, f32)
    y: f32,
    z: f32,
    _pad: f32 = 0.0,

    pub fn init(x: f32, y: f32, z: f32) Self { return .{ .x = x, .y = y, .z = z }; }

    pub fn from_array(arr: [3]f32) Self { return .{ .x = arr[0], .y = arr[1], .z = arr[2] }; }

    pub fn from_array_aligned(arr: *align(16) [4]f32) Self {
        return @bitCast(arr);
    }

    pub fn splat(val: f32) Self { return Self.init( val, val, val ); }

    pub const ZERO = splat(0.0);
    pub const ONE  = splat(1.0);
    pub const X = init(1.0, 0.0, 0.0);
    pub const Y = init(0.0, 1.0, 0.0);
    pub const Z = init(0.0, 0.0, 1.0);
    pub const MIN = splat(std.math.floatMin(f32));
    pub const MAX = splat(std.math.floatMax(f32));

    pub const AXES = [3]Self{X, Y, Z};

    /// Casts the vector components to a @Vector(4, f32) for SIMD
    inline fn v_4(self: anytype) 
    if (@typeInfo(@TypeOf(self)).pointer.is_const) 
        *const @Vector(4, f32) 
    else 
        *@Vector(4, f32)
    {
        return @ptrCast(self);
    }

    /// Rearranges (swizzles/shuffles) components of vector. 
    /// Example: 
    /// ```
    /// var vec = Vector3.init(4, 5, 6);
    /// vec.swizzle("zxy");   // vec is now { 6, 4, 5 }
    /// vec.swizzle("yyy");   // vec is now { 4, 4, 4 }
    /// ```
    pub fn swizzle(self: Self, comptime mask: []const u8) Self {
        if (mask.len != 3) { @compileError("swizzle mask must be length 3"); } 

        comptime var order: [4]i32 = undefined;
        inline for (mask, 0..) |char, i| {
            order[i] = switch (char) {
                'x' => 0, 'y' => 1, 'z' => 2,
                else => @compileError("swizzle components must be x, y or z"),
            };
        }
        order[3] = 0;

        return @bitCast(@shuffle(f32, self.v_4().*, undefined, order)); 
    }

    /// Element-wise
    pub fn add(self: Self, other: Self) Self { return @bitCast(self.v_4().* + other.v_4().*); }

    /// Element-wise
    pub fn sub(self: Self, other: Self) Self { return @bitCast(self.v_4().* - other.v_4().*); }

    /// Element-wise
    pub fn mul(self: Self, other: Self) Self { return @bitCast(self.v_4().* * other.v_4().*); }

    /// Element-wise
    pub fn div(self: Self, other: Self) Self {
        var res: Self = @bitCast(self.v_4().* / other.v_4().*);
        res._pad = 0.0;
        return res;
    }

    pub fn add_scalar(self: Self, scalar: f32) Self { return self.add(Self.splat(scalar)); }

    pub fn sub_scalar(self: Self, scalar: f32) Self { return self.sub(Self.splat(scalar)); }

    pub fn mul_scalar(self: Self, scalar: f32) Self { return self.mul(Self.splat(scalar)); }

    pub fn div_scalar(self: Self, scalar: f32) Self { return self.div(Self.splat(scalar)); }

    /// Element-wise
    pub fn min(self: Self, other: Self) Self { return @bitCast(@min(self.v_4().*, other.v_4().*)); }

    /// Element-wise
    pub fn max(self: Self, other: Self) Self { return @bitCast(@max(self.v_4().*, other.v_4().*)); }

    /// Sum of all elements
    pub fn sum(self: Self) f32 { return @reduce(.Add, self.v_4().*); }

    /// Product of all elements
    pub fn product(self: Self) f32 { return @reduce(.Mul, self.v_4().*); }

    /// Dot product of two vectors
    /// > 0 -> acute angle
    /// = 0 -> 90 degrees
    /// < 0 -> obtuse angle
    pub fn dot(self: Self, other: Self) f32 {
        // ~ 2.85 cycles on ryzen 3800xt
        return self.mul(other).sum();
    }

    /// Magnitude of a
    pub fn len(self: Self) f32 { return @sqrt(self.mul(self).sum()); }

    /// Distance from a->b
    pub fn dist(self: Self, other: Self) f32 { return other.sub(self).len(); }

    pub fn norm(self: Self) Self {
        const a_len = self.len();
        if (a_len == 0.0) { @panic("tried to normalize zero length vector" ); }
        return self.div_scalar(a_len);
    }

    pub fn norm_or_zero(self: Self) Self {
        const a_len = self.len();
        if (a_len == 0.0) { return Self.ZERO; }
        return self.div_scalar(a_len);
    }

    pub fn transform(self: Self, mat: *const Matrix44) Self {
        const cols = mat.v_4x4();
        const v = self.v_4();

        var res = cols[0] * @as(@Vector(4, f32), @splat(v[0]));
        res = @mulAdd(@Vector(4, f32), cols[1], @splat(v[1]), res);
        res = @mulAdd(@Vector(4, f32), cols[2], @splat(v[2]), res);
        res = @mulAdd(@Vector(4, f32), cols[3], @splat(v[3]), res);

        res[3] = 0;
        return @bitCast(res);
    }

    pub fn len_squared(self: Self) f32 { return self.mul(self).sum(); }

    pub fn dist_squared(self: Self, other: Self) f32 { return other.sub(self).len_squared(); }

    pub fn cross(self: Self, other: Self) Self {
        // ~ 9.16 cycles on ryzen 3800xt
        return sub(
            mul(self.swizzle("yzx"), other.swizzle("zxy")),
            mul(self.swizzle("zxy"), other.swizzle("yzx"))
        );
    }

    /// Project a on to b
    /// b's length doesn't matter but must not be zero
    /// **PANIC**s if `b.length() == 0`
    pub fn project(self: Self, other: Self) Self {
        const b_len_squared = other.len_squared();
        if (b_len_squared == 0) { @panic("tried to project onto zero length vector"); }
        return other.mul_scalar( self.dot(other) / b_len_squared );
    }
};
