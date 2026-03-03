const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 1. Define the ymath module. 
    // This is the source-of-truth for your library.
    const mod = b.addModule("ymath", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // 2. Main Library Artifact
    // Users who run 'zig build' will get this.
    const lib = b.addLibrary(.{
        .name = "ymath",
        .root_module = mod,
    });

    b.installArtifact(lib);


    // 3. Test Step
    // Allows 'zig build test' to work.
    const unit_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_unit_tests.step);

    // 4. The 'check' Step
    // This is what ZLS / LSP uses to verify code without building binaries.
    // We use b.addTest because it analyzes all 'test' blocks too.
    const check = b.step("check", "Check if ymath compiles");
    const check_artifact = b.addTest(.{
        .root_module = mod,
    });
    
    // This is the specific fix: you depend on the 'step' of a 
    // compilation artifact (check_artifact) that uses your module.
    check.dependOn(&check_artifact.step);

    // b.default_step.dependOn(b.getInstallStep());
}
