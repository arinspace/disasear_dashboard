class MinHeap:
    def __init__(self):
        self.arr = []

    def push(self, item):
        self.arr.append(item)
        i = len(self.arr) - 1
        while i > 0:
            p = (i - 1) >> 1
            if self.arr[p]["f"] <= self.arr[i]["f"]:
                break
            self.arr[p], self.arr[i] = self.arr[i], self.arr[p]
            i = p

    def pop(self):
        r = self.arr[0]
        last = self.arr.pop()
        if self.arr:
            self.arr[0] = last
            i = 0
            n = len(self.arr)
            while True:
                s = i
                l = i * 2 + 1
                r_idx = i * 2 + 2
                if l < n and self.arr[l]["f"] < self.arr[s]["f"]:
                    s = l
                if r_idx < n and self.arr[r_idx]["f"] < self.arr[s]["f"]:
                    s = r_idx
                if s == i:
                    break
                self.arr[i], self.arr[s] = self.arr[s], self.arr[i]
                i = s
        return r

    def __len__(self):
        return len(self.arr)
