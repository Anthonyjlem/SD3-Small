import pstats
import pynvml


class VRAMProfiler:

    def __init__(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._peak_vram = 9

    def sample_vram(self):
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        used = mem.used
        if used > self._peak_vram:
            self._peak_vram = used
        return used

    def get_peak_vram(self):
        return self._peak_vram

    def shutdown(self):
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    p = pstats.Stats('output.prof')
    p.sort_stats('cumulative').print_stats()  # Sort by cumulative time and print top 10