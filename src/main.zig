const std = @import("std");

const V3 = @import("root.zig").Vector3;
const M44 = @import("root.zig").Matrix44;

const raylib = @cImport({
    @cInclude("raylib.h");
});

pub const FastPrng = struct {
    s: [4]u32,

    pub fn init(seed: u32) FastPrng {
        return .{ .s = .{ seed, seed +% 1, seed +% 2, seed +% 3 } };
    }

    pub fn nextFloat(self: *FastPrng) f32 {
        // Xoshiro128+ algorithm
        const result = self.s[0] +% self.s[3];
        const t = self.s[1] << 9;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = std.math.rotl(u32, self.s[3], 11);

        return @as(f32, @floatFromInt(result)) * 2.3283064e-10 * 2 - 1;
    }
};

fn random_m44(rng: *FastPrng) M44 {
    return M44.init_from_row_major(
    [_]f32{
        rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
        rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
        rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
        rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat()
    });
}

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const count = 30_000_000;
    const data = try allocator.alloc([3]f32, count);
    defer allocator.free(data);

    data[0] = .{ 1.0, 2.0, 3.0 };
    
    var rng = FastPrng.init(0);

    for (data) |*item| {
        item.* = .{ rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), };
    }

    var acc: f32 = 0.0;
    var iter = std.mem.window([3]f32, data, 3, 3);

    var t: std.Io.Threaded = .init_single_threaded;
    var start = std.Io.Clock.real.now(t.io());

    while (iter.next()) |chunk| {
        var xx = V3.from_array(chunk[0]).norm();
        var yy = V3.from_array(chunk[1]).norm();
        const pp = V3.from_array(chunk[2]).norm();

        xx = xx.add(V3.splat(0.1));
        yy = yy.sub(V3.splat(0.1));
        var zz = V3.cross(xx, yy).project(pp);
        acc += zz.len();
    }

    var end = std.Io.Clock.real.now(t.io());
    var dur: f32 = @floatFromInt(start.durationTo(end).nanoseconds);
    dur /= std.time.ns_per_s;
    std.debug.print("acc={}\n", .{acc});
    std.debug.print("{} secs\n", .{dur});
    std.debug.print("{} per sec\n", .{count / 3 / dur});
    std.debug.print("{} per frame @ 60fps\n", .{((count / 3) / dur) / 60});

    const x = V3.X;
    const other = V3.ONE.project(x);
    std.debug.print("{any}\n", .{other});

    // const m1 = random_m44(&rng);
    // const m2 = random_m44(&rng);
    // const m3 = m1.add(m2);
    // std.debug.print("\n", .{});
    // m3.print();
    // std.debug.print("\n", .{});
    // m1.matmul(m2).print();
    // std.debug.print("\n", .{});
    // m3.add(M44.IDENTITY.mul_scalar(5)).print();
    
    const countm = 30_000_000;
    const datam = try allocator.alloc(M44, countm);

    for (datam) |*item| {
        item.* = random_m44(&rng);
    }

    var accm: f32 = 0.0;
    var iterm = std.mem.window(M44, datam, 3, 3);

    start = std.Io.Clock.real.now(t.io());

    while (iterm.next()) |chunk| {
        const m_1 = chunk[0];
        const m_2 = chunk[1];
        const m_3 = chunk[2];
        const m_4 = m_1.matmul(m_2).mul(m_3.transpose());
        accm += m_4.reduce_sum();
    }

    end = std.Io.Clock.real.now(t.io());
    dur = @floatFromInt(start.durationTo(end).nanoseconds);
    dur /= std.time.ns_per_s;
    std.debug.print("accm={}\n", .{accm});
    std.debug.print("{} secs\n", .{dur});
    std.debug.print("{} per sec\n", .{countm / 3 / dur});
    std.debug.print("{} per frame @ 60fps\n", .{((countm / 3) / dur) / 60});

    std.debug.print("\n", .{});
    const some_vec = V3.X;
    for (0..17) |i| {
        const step: f32 = @floatFromInt(i);
        const angle: f32 = (step / 16) * std.math.pi * 2;
        const some_mat = M44.rotation_z(angle);
        std.debug.print("{any}\n", .{some_vec.transform(&some_mat)});
    }
}
