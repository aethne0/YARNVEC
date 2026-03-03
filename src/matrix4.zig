const std = @import("std");

pub fn Mat4(comptime FType: type) type {
    return extern struct {
        const Self = @This();

        // Layout wise this is a `@Vector(16, f32)` or `[4]@Vector(4, f32)`
        v00: FType align(64), v10: FType, v20: FType, v30: FType,
        v01: FType,           v11: FType, v21: FType, v31: FType,
        v02: FType,           v12: FType, v22: FType, v32: FType,
        v03: FType,           v13: FType, v23: FType, v33: FType,

        pub fn init(vals: [16]FType) Self {
            var mat: Self = undefined;
            inline for (0..16) |i| {
                mat.v_16()[i] = vals[i];
            }
            return mat;
        }

        pub fn init_from_aligned(vals: *align(16) [16]FType) Self {
            return @bitCast(vals);
        }

        pub inline fn splat(val: FType) Self {
            const res: @Vector(16, FType) = @splat(val);
            return @bitCast(res);
        }

        pub const IDENTITY = Self.init(
        [_]FType{
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        });

        pub const ZERO = Self.splat(0);
        pub const ONE = Self.splat(1);

        /// Cast to 4 packed 4-element FType vectors
        inline fn v_4x4(self: anytype) 
        if (@typeInfo(@TypeOf(self)).pointer.is_const) 
            *const [4]@Vector(4, FType) 
        else 
            *[4]@Vector(4, FType)
        {
            return @ptrCast(self);
        }

        /// Cast to 16-element FType vector
        inline fn v_16(self: anytype) 
        if (@typeInfo(@TypeOf(self)).pointer.is_const) 
            *const @Vector(16, FType) 
        else 
            *@Vector(16, FType)
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

        pub fn add_scalar(self: Self, scalar: FType) Self {
            return add(self, splat(scalar));
        }

        pub fn sub_scalar(self: Self, scalar: FType) Self {
            return sub(self, splat(scalar));
        }

        pub fn mul_scalar(self: Self, scalar: FType) Self {
            return mul(self, splat(scalar));
        }

        pub fn div_scalar(self: Self, scalar: FType) Self {
            return div(self, splat(scalar));
        }

        pub fn reduce_sum(self: Self) FType {
            return @reduce(.Add, self.v_16().*);
        }

        pub fn transpose(self: Self) Self {
            return @bitCast(
                @shuffle(FType, self.v_16().*, undefined, 
                    [_]i32{
                    0,  4,  8, 12,
                    1,  5,  9, 13,
                    2,  6, 10, 14,
                    3,  7, 11, 15,
                })
            );
        }

        pub fn init_from_row_major(vals: [16]FType) Self {
            return init(vals).transpose();
        }

        pub fn rotation_x(angle: FType) Self {
            var mat = IDENTITY;
            mat.v_16()[ 5] =  @cos(angle); mat.v_16()[ 6] = -@sin(angle);
            mat.v_16()[ 9] =  @sin(angle); mat.v_16()[10] =  @cos(angle);
            return mat;
        }

        pub fn rotation_y(angle: FType) Self {
            var mat = IDENTITY;
            mat.v_16()[ 0] =  @cos(angle); mat.v_16()[ 2] =  @sin(angle);
            mat.v_16()[ 8] = -@sin(angle); mat.v_16()[10] =  @cos(angle);
            return mat;
        }

        pub fn rotation_z(angle: FType) Self {
            var mat = IDENTITY;
            mat.v_16()[ 0] =  @cos(angle); mat.v_16()[ 1] = -@sin(angle);
            mat.v_16()[ 4] =  @sin(angle); mat.v_16()[ 5] =  @cos(angle);
            return mat;
        }

        pub fn matmul(self: Self, other: Self) Self {
            var result: Self = undefined;
            const a = self.v_4x4();
            const b = other.v_4x4();

            inline for (0..4) |i| {
                const col = b[i];
                
                var sum: @Vector(4, FType) = a[0] * @as(@Vector(4, FType), @splat(col[0]));
                sum = @mulAdd(@Vector(4, FType), a[1], @splat(col[1]), sum);
                sum = @mulAdd(@Vector(4, FType), a[2], @splat(col[2]), sum);
                sum = @mulAdd(@Vector(4, FType), a[3], @splat(col[3]), sum);
                
                result.v_4x4()[i] = sum;
            }
            
            return result;
        }
    };
}
