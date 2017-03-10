import "regent"

local c = regentlib.c

local DIAMETER = 5

struct Queue {
    entries : int3d[DIAMETER * DIAMETER];
    curr : uint32;
    length : uint32;
}

terra Queue:init()
    self.curr = 0
    self.length = 0
end

terra Queue:dequeue() 
    if self.length <= 0 then
        -- c.printf("Dequeueing point from queue that is empty\n")
    end
    
    var toRet = self.entries[self.curr]
    self.length = self.length - 1
    self.curr = (self.curr + 1) % (DIAMETER * DIAMETER)
    return toRet
end

terra Queue:enqueue(p : int3d) 
    if self.length >= DIAMETER * DIAMETER then
        -- c.printf("Adding point to queue that is full\n")
    end
    
    self.entries[(self.curr + self.length) % (DIAMETER * DIAMETER)] = p
    self.length = self.length + 1
end

terra Queue:empty() 
    return self.length <= 0
end

terra Queue:clear() 
    self.length = 0
end

return Queue