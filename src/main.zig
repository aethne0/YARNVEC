const std = @import("std");

const V3 = @import("root.zig").vec.Vector3;
const M44 = @import("root.zig").mat.Matrix44;

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

        return @as(f32, @floatFromInt(result)) * 2.3283064e-10;
    }
};

fn random_m44(rng: *FastPrng) M44 {
    return M44.init([_]f32{
        rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), 
        rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), 
        rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), 
        rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
    });
}

fn rdtsc() u32 {
    return asm volatile ("rdtsc" : [low] "={eax}" (-> u32));
}

pub fn main(_: std.process.Init) !void {
    const count = 16_000;

    // dot
    {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        defer _ = gpa.deinit();
        const allocator = gpa.allocator();

        const data = try allocator.alloc(V3, count);
        defer allocator.free(data);

        var rng = FastPrng.init(0);

        for (data) |*item| {
            item.* = V3.init(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        }

        var acc: f32 = 0.0;
        var iter = std.mem.window(V3, data, 2, 2);

        var t: std.Io.Threaded = .init_single_threaded;
        var start = std.Io.Clock.real.now(t.io());

        std.debug.print("dot...\n", .{});
        const start_cycles = rdtsc();

        while (iter.next()) |chunk| {
            acc += V3.dot(chunk[0], chunk[1]);
        }

        const end_cycles = rdtsc();

        const end = std.Io.Clock.real.now(t.io());
        const dur_ns: f32 = @floatFromInt(start.durationTo(end).nanoseconds);
        const dur_us = dur_ns / std.time.ns_per_us;
        const perop = dur_ns / (count / 2);

        std.debug.print("RESULT: {}\n", .{acc});
        std.debug.print("{:.2} μs\n", .{dur_us});
        std.debug.print("{:.2} ns per op\n", .{perop});
        const float_op_cnt: f64 = @floatFromInt(end_cycles - start_cycles);
        std.debug.print("{} cycles\n", .{ float_op_cnt });
        std.debug.print("{:.2} cycles per op\n", .{ float_op_cnt / (count / 2)});
    }

    // cross
    {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        defer _ = gpa.deinit();
        const allocator = gpa.allocator();

        const data = try allocator.alloc(V3, count);
        defer allocator.free(data);

        var rng = FastPrng.init(0);

        for (data) |*item| {
            item.* = V3.init(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        }

        var acc = V3.ZERO;

        var t: std.Io.Threaded = .init_single_threaded;
        var start = std.Io.Clock.real.now(t.io());

        std.debug.print("\ncross...\n", .{});
        const start_cycles = rdtsc();

        for (data) |v| {
            acc = acc.add(v);
        }

        const end_cycles = rdtsc();

        const end = std.Io.Clock.real.now(t.io());
        const dur_ns: f32 = @floatFromInt(start.durationTo(end).nanoseconds);
        const dur_us = dur_ns / std.time.ns_per_us;
        const perop = dur_ns / count;


        std.debug.print("RESULT: {any}\n", .{acc});
        std.debug.print("{:.2} μs\n", .{dur_us});
        std.debug.print("{:.2} ns per op\n", .{perop});
        const float_op_cnt: f64 = @floatFromInt(end_cycles - start_cycles);
        std.debug.print("{} cycles\n", .{ float_op_cnt });
        std.debug.print("{:.2} cycles per op\n", .{ float_op_cnt / count});
    }

    // mat
    {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        defer _ = gpa.deinit();
        const allocator = gpa.allocator();

        const data = try allocator.alloc(M44, count);
        defer allocator.free(data);

        var rng = FastPrng.init(0);

        for (data) |*item| {
            item.* = random_m44(&rng);
        }

        var acc_mat = M44.ONE;

        var t: std.Io.Threaded = .init_single_threaded;
        var start = std.Io.Clock.real.now(t.io());

        std.debug.print("\nadd->transpose...\n", .{});
        const start_cycles = rdtsc();

        for (data) |mat| {
            acc_mat = acc_mat.add(mat);
            acc_mat = acc_mat.transpose();
        }

        const end_cycles = rdtsc();

        const end = std.Io.Clock.real.now(t.io());
        const dur_ns: f32 = @floatFromInt(start.durationTo(end).nanoseconds);
        const dur_us = dur_ns / std.time.ns_per_us;
        const perop = dur_ns / count;

        std.debug.print("RESULT:\n", .{}); 
        acc_mat.print();
        std.debug.print("{:.2} μs\n", .{dur_us});
        std.debug.print("{:.2} ns per op\n", .{perop});
        const float_op_cnt: f64 = @floatFromInt(end_cycles - start_cycles);
        std.debug.print("{} cycles\n", .{ float_op_cnt });
        std.debug.print("{:.2} cycles per op\n", .{ float_op_cnt / count});
    }
}
