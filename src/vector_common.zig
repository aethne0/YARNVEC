const std = @import("std");
// -O ReleaseFast -mcpu znver5
// -O ReleaseFast -target aarch64-linux-gnu -mcpu cortex_a55
// -C opt-level=3 -C target-cpu=znver5
// -C opt-level=3 -C target-cpu=cortex_a55

/// Common implementations of methods for Vec2, Vec3A, Vec4
pub fn VectorAlignedCommon(comptime FType: type, Dims: comptime_int, OurType: type) type {
    // note: OurType can be derived from Dims, of course, but it means that another 
    // equivalent but anonymous type will be created that isn't linked to the actual type of 
    // the implementation calling this - this makes it so my ZLS won't give us any info about
    // these shared functions when calling them from the actual concrete implementations (Vec3A, Vec4, Vec2)
    //
    // The code still compiles and works though, so this may be a ZLS shortcoming (not that 
    // there aren't reasons to avoid having your LSP do 10 layers of monomorphization to give
    // you information).
    const BitsType = switch (FType) { 
        f32 => u128, f64 => u256,
        else => @compileError("FType must be f32 | f64"),
    };


    const VType = if (Dims == 2) @Vector(2, FType) else @Vector(4, FType);

    const Our2Type = @import("vector_2.zig").Vector2(FType);
    const Our3Type = @import("vector_3a.zig").Vector3A(FType);
    const Our4Type = @import("vector_4.zig").Vector4(FType);

    return struct {
        const Self = OurType;

        /// Initializes vector with `scalar` for all elements
        /// Example `Vec3A.splat(2)` -> `{ 2, 2, 2 }`
        ///         `Vec2.splat(3)` -> `{ 3, 3 }`
        pub inline fn splat(scalar: FType) Self {
            return @bitCast(@as(VType, @splat(scalar)));
        }

        /// Vector initialized with all values 0
        pub const ZERO      = splat(0);
        /// Vector initialized with all values 1
        pub const ONE       = splat(1);
        /// Vector initialized with all values -1
        pub const NEG_ONE   = splat(-1);
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

        pub inline fn as_vec(self: Self) VType {
            return @bitCast(self);
        }

        /// Element-wise add
        pub fn add(self: Self, other: Self) Self {
            return @bitCast( as_vec(self) + as_vec(other) );
        }

        /// Element-wise subtract
        pub fn sub(self: Self, other: Self) Self {
            return @bitCast( as_vec(self) - as_vec(other) );
        }

        /// Element-wise multiply
        pub fn mul(self: Self, other: Self) Self {
            return @bitCast(as_vec(self) * as_vec(other));
        }

        /// Element-wise divide
        pub fn div(self: Self, other: Self) Self {
            return @bitCast(as_vec(self) / as_vec(other));
        }

        /// Element-wise (self * multiplier) + addend. 
        /// Will use fused multiply add optimizations if available.
        pub fn mulAdd(self: Self, multiplier: Self, addend: Self) Self {
            return @bitCast(
                @mulAdd(VType,
                    as_vec(self),
                    multiplier.as_vec(),
                    addend.as_vec()
                )
            );
        }

        /// Adds scalar to each element
        pub fn addScalar(self: Self, scalar: FType) Self {
            return self.add(splat(scalar));
        }

        /// Subtracts scalar from each element
        pub fn subScalar(self: Self, scalar: FType) Self {
            return self.sub(splat(scalar));
        }

        /// Multiplies each element by scalar
        pub fn mulByScalar(self: Self, scalar: FType) Self {
            return self.mul(splat(scalar));
        }

        /// Divides each element by scalar
        pub fn divByScalar(self: Self, scalar: FType) Self {
            return div(self, splat(scalar));
        }

        /// Element-wise negate
        pub fn neg(self: Self) Self {
            return mulByScalar(self, -1);
        }

        /// Element-wise ceil
        pub fn ceil(self: Self) Self {
            return @bitCast(@ceil(as_vec(self)));
        }

        /// Element-wise round
        pub fn round(self: Self) Self {
            return @bitCast(@round(as_vec(self)));
        }

        /// Element-wise floor
        pub fn floor(self: Self) Self {
            return @bitCast(@floor(as_vec(self)));
        }

        /// Element-wise sin
        pub fn sin(self: Self) FType {
            return @bitCast(@sin(as_vec(self)));
        }

        /// Element-wise cos
        pub fn cos(self: Self) FType {
            return @bitCast(@cos(as_vec(self)));
        }

        /// Element-wise natural logarithm
        pub fn ln(self: Self) FType {
            return @bitCast(@log(as_vec(self)));
        }

        /// Element-wise base-2 logarithm
        pub fn log2(self: Self) FType {
            return @bitCast(@log2(as_vec(self)));
        }

        /// Element-wise e^self
        pub fn exp(self: Self) FType {
            return @bitCast(@exp(as_vec(self)));
        }

        /// Element-wise 2^self
        pub fn exp2(self: Self) FType {
            return @bitCast(@exp2(as_vec(self)));
        }

        /// Element-wise reciprocal (1/x)
        pub fn recip(self: Self) FType {
            return ONE.div(self);
        }

        /// Element-wise sqrt
        pub fn sqrt(self: Self) FType {
            return @bitCast(@sqrt(as_vec(self)));
        }

        /// Approximate reciprocal square root of each element, only faster on arch that supports it,
        /// and usually only for f32.
        pub fn recipSqrtFast(self: Self) Self {
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
            const one: VType = @splat(1);
            return @bitCast(one / @sqrt(as_vec(self)));
        }

        /// Element-wise absolute value
        pub fn abs(self: Self) Self {
            return @bitCast(@abs(as_vec(self)));
        }

        /// Element-wise max operation
        pub fn max(self: Self, other: Self) Self {
            return @bitCast(@max(as_vec(self), as_vec(other)));
        }

        /// Element-wise min operation
        pub fn min(self: Self, other: Self) Self {
            return @bitCast(@min(as_vec(self), as_vec(other)));
        }

        /// Element-wise clamp from [0, 1]
        pub fn saturate(self: Self) Self {
            // todo: perf maybe
            return self.clamp(ZERO, ONE);
        }

        /// Square of the length of the vector, skips computing sqrt
        pub fn lengthSquared(self: Self) FType {
            return self.mul(self).sum();
        }

        /// Length of the vector
        pub fn length(self: Self) FType {
            return @sqrt(lengthSquared(self)); // i BELIEVE in llvm inlining
        }

        /// Reciprocal of the length of the vector (1 / length)
        pub fn lengthRecip(self: Self) FType {
            return ONE.div(self.length());
        }

        /// Sets length of vector to `upper_bound` if it is exceeding `upper_bound`, otherwise does nothing.
        /// Does not affect the direction of the vector. `upper_bound` must be zero or positive.
        pub fn clampLengthMax(self: Self, upper_bound: FType) Self {
            if (upper_bound < 0) std.debug.panic("clamp_length_max upper_bound must be >= 0", .{});
            if (upper_bound == 0) return ZERO;

            const sqr_length = lengthSquared(self);
            const sqr_upper_bound = upper_bound * upper_bound;

            if (sqr_length > sqr_upper_bound) {
                return mulByScalar(self, upper_bound / @sqrt(sqr_length));
            } else {
                return self;
            }
        }

        /// Sets length of vector to `lower_bound` if it is less-than `lower_bound`, otherwise does nothing.
        /// Does not affect the direction of the vector. `lower_bound` must be zero or positive.
        /// PANICS if length of `self` is 0!
        pub fn clampLengthMin(self: Self, lower_bound: FType) Self {
            if (lower_bound < 0) std.debug.panic("clamp_length_min lower_bound must be >= 0", .{});

            const sqr_length = lengthSquared(self);
            if (sqr_length == 0) std.debug.panic("tried to clamp_min_length zero length vector", .{});
            const sqr_lower_bound = lower_bound * lower_bound;

            if (sqr_length < sqr_lower_bound) {
                return mulByScalar(self, lower_bound / @sqrt(sqr_length));
            } else {
                return self;
            }
        }

        /// Sets length of vector to `lower_bound` if it is less-than `lower_bound`, and sets
        /// length to `upper_bound` if its greater-than `upper_bound`. If it is already within
        /// this inclusive range this will have no effect.
        /// Does not affect the direction of the vector.
        /// `lower_bound` and `upper_bound` must be zero or positive.
        pub fn clampLength(self: Self, lower_bound: FType, upper_bound: FType) Self {
            if (lower_bound > upper_bound) std.debug.panic("lower_ bound must be <= upper_bound", .{});
            const max_clamped = clampLengthMax(self, upper_bound);
            return clampLengthMin(max_clamped, lower_bound);
        }

        /// Scales vector so that length is `len`, does not affect direction. `len` must be >= 0;
        pub fn setLength(self: Self, len: FType) Self {
            if (len < 0) std.debug.panic("length cannot be < 0", .{});
            return self.mul_scalar(len / self.length_recip());
        }

        /// Distance from self -> other
        /// When called as a method you can read this as "distanceTo"
        pub fn distanceSquared(self: Self, other: Self) FType {
            const diff = other.sub(self);
            return diff.mul(diff).sum();
        }

        /// Distance from self -> other
        /// When called as a method you can read this as "distanceTo"
        pub fn distance(self: Self, other: Self) FType {
            return @sqrt(distance(self, other));
        }

        /// One over the distance of self -> other
        /// When called as a method you can read this as "distanceRecipTo"
        pub fn distanceReciprocal(self: Self, other: Self) FType {
            // todo: perf
            return ONE.div(distance(self, other));
        }

        /// Scales vector such that it is length=1, does not affect direction.
        /// PANICS if length is zero!
        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len == 0) std.debug.panic("tried to normalize zero length vector", .{});
            return divByScalar(self, len);
        }

        /// Scales vector such that it is length=1, does not affect direction.
        /// Also returns the length, which was computed as a side-effect.
        /// PANICS if length is zero!
        pub fn normalizeAndLength(self: Self) struct { vec: Self, length: FType } {
            const len = self.length();
            if (len == 0) std.debug.panic("tried to normalize zero length vector", .{});
            return .{ .vec = divByScalar(self, len), .length = len };
        }

        /// Scales vector such that it is length=1, does not affect direction.
        /// Returns ZERO vector if length is zero!
        pub fn normalizeOrZero(self: Self) Self {
            const len = self.length();
            if (len == 0) return self.ZERO;
            return self.div_scalar(len);
        }

        pub fn isNormalized(self: Self) Self {
            return @abs(self.length_squared() - 1) <= std.math.floatEpsAt(FType, 1);
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
        
        /// Example:
        /// ```zig
        /// const a = Vec3A.init(1, 2, 3);
        /// _ = a.swizzle("zzy"); // -> { 3, 3, 2 }
        /// ```
        pub fn swizzle(self: Self, comptime mask: []const u8) Self {
            if (comptime mask.len != Dims) @compileError("swizzle mask must be length equal to dimensions");

            const order_len = if (comptime Dims == 2) 2 else 4;

            comptime var order: [order_len]isize = undefined;
            inline for (mask, 0..) |char, i| {
                order[i] = switch(comptime char) {
                    'x' => 0, 'y' => 1, 'z' => 2, 'w' => 3,
                    else => @compileError("invalid axis label"),
                };
                if (comptime order[i] + 1 > Dims) @compileError("axis label out of range or invalid");
            }

            if (comptime Dims == 3) order[3] = 0; // dont-care

            return @bitCast(@shuffle(FType, as_vec(self), undefined, order));
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
        pub fn swizzleResize(self: Self, comptime mask: []const u8) 
            switch (mask.len) {
                2 => Our2Type, 3 => Our3Type, 4 => Our4Type,
                else => @compileError("`swizzle_and_resize` mask length must be 2, 3 or 4"),
            }
        {
            const order_len = if (mask.len == 2) 2 else 4; // keep pad if making a vec3

            comptime var order: [order_len]isize = undefined;
            inline for (mask, 0..) |char, i| {
                order[i] = switch(comptime char) {
                    'x' => 0, 'y' => 1, 'z' => 2, 'w' => 3,
                    else => @compileError("invalid axis label (must be x, y, z or 0)"),
                };
                if (comptime order[i] + 1 > Dims) @compileError("axis label out of range or invalid");
            }

            if (comptime mask.len == 3) order[3] = 0; // dont-care

            return @bitCast(@shuffle(FType, as_vec(self), undefined, order));
        }

        // *********************************************************************
        // *********************************************************************
        //  Only for length 2 | 4
        // *********************************************************************
        // *********************************************************************

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

        /// Dot product
        pub fn dot(self: Self, other: Self) FType {
            return sum(mul(self, other));
        }

        /// min of all elements
        pub fn minElement(self: Self) FType {
            return @reduce(.Min, self.as_vec_3());
        }

        /// max of all elements
        pub fn maxElement(self: Self) FType {
            return @reduce(.Max, self.as_vec_3());
        }

        /// Sum of all elements
        pub fn sum(self: Self) FType {
            return @reduce(.Add, as_vec(self));
        }

        /// Element-wise bitwise equality
        pub fn eq(self: Self, other: Self) bool {
            // reference:
            // const result = erch.x86._mm_add_ps(as_vec(self), as_vec(other));
            // return 0 != arch.x86._mm_testz_ps(result, result);
            return @as(BitsType, @bitCast(as_vec(self))) 
                == @as(BitsType, @bitCast(as_vec(other)));
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
            var res = as_vec(self);
            res = @min(res, as_vec(splat(upper_bound)));
            res = @max(res, as_vec(splat(lower_bound)));
            return @bitCast(res);
        }

        /// Element-wise clamp
        /// Note: This is not gauranteed to observe IEEE 754.
        /// on x86_64 it will probably emit `vmaxps`/`vminps` which do not.
        pub fn clamp(self: Self, lower_bound: Self, upper_bound: Self) Self {
            if (!@reduce(.And, upper_bound.as_vec() >= lower_bound.as_vec()))
                std.debug.panic(
                    \\ called clamp with lower_bound > upper_bound (for one or more elements):
                    \\ lower: {any}, upper: {any}
                    , .{lower_bound, upper_bound});
            return self.min(upper_bound).max(lower_bound);
        }
    };
}
