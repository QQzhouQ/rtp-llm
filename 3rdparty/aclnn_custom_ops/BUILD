package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(
        [
            "**/*.cmake",
            "**/*.cmake.in",
            "cmake/**/*",
            "utils/**/*",         # CANN 工具库（tiling/proto headers）
            "*/**/*",             # 自动捕获所有算子子目录（apply_top_k_top_p_custom/、moe_gating_top_k/ 等）
        ],
    ) + [
        "CMakeLists.txt",
        "build.sh",
        "build_for_bazel.sh",
    ],
)

genrule(
    name = "build_aclnn_ops",
    srcs = [":all_srcs"],
    outs = [
        "lib/libcust_opapi.so",
        "lib/libcust_opsproto_rt2.0.so",
        "lib/libcust_opmaster_rt2.0.so",
    ],
    # 算子名称列表：单算子用 "apply_top_k_top_p_custom"，多算子用分号分隔 "op_a;op_b;op_c"
    cmd = "bash $(location build_for_bazel.sh) " +
          "$(RULEDIR) 'apply_top_k_top_p_custom' ascend910b",
    tools = ["build_for_bazel.sh"],
    # 仅在 Ascend 平台执行；其他平台跳过
    target_compatible_with = select({
        "@//:using_ascend": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    tags = ["local"],   # 需要访问 CANN 环境，不禁用 sandbox
)

cc_library(
    name = "aclnn_custom_ops",
    data = [
        "lib/libcust_opapi.so",
        "lib/libcust_opsproto_rt2.0.so",
        "lib/libcust_opmaster_rt2.0.so",
    ],
    deps = [
        ":build_aclnn_ops",  # 触发 genrule 执行，生成 lib/*.so
        "@local_config_ascend//ascend:ascend_headers",
    ],
)