const std = @import("std");
const vector_common = @import("vector_common.zig");

pub fn Vector2(comptime FType: type) type {
    if (FType != f32 and FType != f64) @compileError("fType must be f32 | f64");

    return extern struct {
        const Self = @This();
        const shared = vector_common.VectorAlignedCommon(FType, 2, Self);

        x:      FType align(@sizeOf(FType) * 2),
        y:      FType,

        /// Initialize struct with values
        /// Example: `const some_vec = Vec3A.init(.{1, 2, 3});`
        pub fn init(x: FType, y: FType) Self {
            return .{ .x = x, .y = y };
        }

        const as_vec = shared.as_vec;

        pub const splat = shared.splat;

        /// X-direction unit vector
        pub const X         = init(1, 0);
        /// Y-direction unit vector
        pub const Y         = init(0, 1);
        /// negative X-direction unit vector
        pub const NEG_X     = init(-1, 0);
        /// negative Y-direction unit vector
        pub const NEG_Y     = init(0, -1);
        /// array of all positive unit vectors
        pub const AXES      = [_]Self{ X, Y };

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
        pub const addScalar  = shared.addScalar;
        pub const subScalar  = shared.subScalar;
        pub const mulByScalar  = shared.mulByScalar;
        pub const divByScalar = shared.divByScalar;
        pub const mulAdd = shared.mulAdd;
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

        pub const product  = shared.product;
        pub const minElement  = shared.minElement;
        pub const maxElement  = shared.maxElement;
        pub const sum  = shared.sum;
        pub const eq = shared.eq;
        pub const clampByScalars = shared.clampByScalars;
        pub const clamp = shared.clamp;
        pub const dot = shared.dot;

    };
}

/// 2-dimensional f32 vector
pub const Vec2 = Vector2(f32);
/// 2-dimensional f64 vector
pub const Vec2f64 = Vector2(f64);


// ▄▄▄▄▄▄▄▄ ..▄▄ · ▄▄▄▄▄.▄▄ · 
// •██  ▀▄.▀·▐█ ▀. •██  ▐█ ▀. 
//  ▐█.▪▐▀▀▪▄▄▀▀▀█▄ ▐█.▪▄▀▀▀█▄
//  ▐█▌·▐█▄▄▌▐█▄▪▐█ ▐█▌·▐█▄▪▐█
//  ▀▀▀  ▀▀▀  ▀▀▀▀  ▀▀▀  ▀▀▀▀ 

fn Vec2Tests (FType: type) type {
    const t = std.testing;

    return struct  {
        const VType = Vector2(FType);

        test "stuff" {
            const some = VType.init(1, 2);
            try t.expectEqual(1, some.x);
            try t.expectEqual(2, some.y);
        }
    };
}

comptime {
    _ = Vec2Tests(f32);
    _ = Vec2Tests(f64);
}

