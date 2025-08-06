class MemoryBank:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memory = []

    def store(self, x):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(x)

    def get_recent(self, n=10):
        return self.memory[-n:]
