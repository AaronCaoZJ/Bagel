import csv
import sys

def parse_ncu_csv(filename, peak_bandwidth_GBs=None):
    dram_read = 0
    dram_write = 0
    pct_of_peak = []

    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            metric = row[1].strip()
            value = row[3].strip()

            try:
                if metric == "dram__bytes_read.sum":
                    dram_read += float(value)
                elif metric == "dram__bytes_write.sum":
                    dram_write += float(value)
                elif metric == "dram__throughput.avg.pct_of_peak_sustained_elapsed":
                    pct_of_peak.append(float(value))
            except ValueError:
                continue

    total_bytes = dram_read + dram_write
    total_GB = total_bytes / 1e9

    print(f"Total DRAM Read : {dram_read/1e9:.3f} GB")
    print(f"Total DRAM Write: {dram_write/1e9:.3f} GB")
    print(f"Total DRAM I/O  : {total_GB:.3f} GB")

    if peak_bandwidth_GBs:
        # 平均时间利用率估计：总字节 / 峰值带宽 = 最低执行时间下限
        min_exec_time = total_GB / peak_bandwidth_GBs
        print(f"Peak Bandwidth : {peak_bandwidth_GBs:.1f} GB/s")
        print(f"Lower bound runtime (bandwidth-limited): {min_exec_time:.6f} s")

    if pct_of_peak:
        avg_pct = sum(pct_of_peak) / len(pct_of_peak)
        print(f"Avg. % of Peak Sustained BW: {avg_pct:.2f} %")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_ncu_bandwidth.py profile_output.csv [peak_bandwidth_GBs]")
        sys.exit(1)

    filename = sys.argv[1]
    peak = float(sys.argv[2]) if len(sys.argv) > 2 else None
    parse_ncu_csv(filename, peak)
