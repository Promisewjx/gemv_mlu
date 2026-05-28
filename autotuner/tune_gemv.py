import datetime
import subprocess
import time
import numpy as np
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt
import re
import os

GEMV_bds = [
    {'name': 'sram_rows', 'type': 'discrete', 'domain': [16, 32, 64, 128, 256]},
    {'name': 'nram_floats', 'type': 'discrete', 'domain': [64, 128, 256, 512]},
    {'name': 'pipeline_bufs', 'type': 'discrete', 'domain': [1, 2]},
    {'name': 'unroll_factor', 'type': 'discrete', 'domain': [1, 2, 4, 8]}
]

def run_gemv_kernel(parameters):
    """
    接收一组参数，运行C++启动器，并解析性能结果 (GFLOPS)。
    """
    params = parameters[0]
    sram_rows_val = int(params[0])
    nram_floats_val = int(params[1])
    pipeline_bufs_val = int(params[2])
    unroll_factor_val = int(params[3])

    try:
        # --- 构建命令行 ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        launcher_path = os.path.join(project_root, 'build', 'bin', 'gemv_bo_launcher')

        # 检查启动器是否存在
        if not os.path.exists(launcher_path):
            print(f"[ERROR] Launcher not found at: {launcher_path}")
            print("Please build the GEMV project first: mkdir -p build && cd build && cmake -DHAVE_MLU=ON .. && make -j")
            return 0.0 # 返回一个差的分数

        cmd = [
            launcher_path,
            '--sram_rows', str(sram_rows_val),
            '--nram_floats', str(nram_floats_val),
            '--pipeline_bufs', str(pipeline_bufs_val),
            '--unroll_factor', str(unroll_factor_val)
        ]

        print(f"\n[INFO] Testing with params: SRAM_ROWS={sram_rows_val}, NRAM_FLOATS={nram_floats_val}, PIPELINE_BUFS={pipeline_bufs_val}, UNROLL_FACTOR={unroll_factor_val}")

        # --- 执行 C++ 程序并捕获输出 ---
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        # --- 解析性能得分 ---
        if process.returncode == 0:
            # 使用正则表达式从最后一行匹配 "GFLOPS: xxx.xx"
            match = re.search(r'GFLOPS:\s*(\d+\.?\d*)', stdout)
            if match:
                score = float(match.group(1))
                print(f"[SUCCESS] GFLOPS = {score:.6f}")
                return score
            else:
                print("[WARNING] Could not parse GFLOPS from output.")
                print("STDOUT:", stdout)
                return 0.0
        else:
            print(f"[ERROR] C++ launcher failed with return code {process.returncode}")
            print("STDERR:", stderr)
            return 0.0 # 返回一个差的分数

    except Exception as e:
        print(f"[CRITICAL] An exception occurred: {e}")
        return 0.0

# ============================================================================
# 3. 配置并运行贝叶斯优化器
# ============================================================================
if __name__ == "__main__":
    # 定义优化器
    optimizer = BayesianOptimization(
        f=run_gemv_kernel,
        domain=GEMV_bds,
        model_type='GP',          # 使用高斯过程作为模型
        acquisition_type='EI',    # 使用期望提升(Expected Improvement)作为采集函数
        acquisition_jitter=0.05,
        exact_feval=True,
        maximize=True             
    )

    # 设置迭代次数
    iteration_num = 50 
    print(f"--- Starting Bayesian Optimization for {iteration_num} iterations ---")
    optimizer.run_optimization(max_iter=iteration_num, verbosity=True)
    print("--- Bayesian Optimization Finished ---")


    # --- 结果展示 ---
    print("\n=============================================")
    print("           Optimization Results")
    print("=============================================")
    # GPyOpt 默认是最小化，所以最大化的最优值是 -optimizer.fx_opt
    print(f"Optimized GFLOPS: {-optimizer.fx_opt:.6f}")
    
    # 打印最优参数
    best_params = {
        'SRAM_BLOCK_ROWS': int(optimizer.x_opt[0]),
        'NRAM_CHUNK_FLOATS': int(optimizer.x_opt[1]),
        'PIPELINE_BUFFERS': int(optimizer.x_opt[2]),
        'UNROLL_FACTOR': int(optimizer.x_opt[3])
    }
    print("Best Parameters Found:")
    for name, value in best_params.items():
        print(f"  - {name}: {value}")
    print("=============================================\n")

    # --- 绘制并保存收敛图 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_name = os.path.join(script_dir, f"gemv_bo_convergence_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    optimizer.plot_convergence()
    plt.title('Convergence Plot of GEMV Optimization')
    plt.ylabel('Best GFLOPS found')
    plt.tight_layout()
    plt.savefig(fig_name)
    print(f"Convergence plot saved to: {fig_name}")
    plt.show()