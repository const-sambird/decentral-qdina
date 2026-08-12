import random
from collections import namedtuple

# Define a transition tuple to store all relevant information for a single step.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_action_mask'))


class ReplayMemory:
    """
    A fixed-capacity experience replay buffer implemented as a circular list.

    This implementation uses a pre-allocated list to store transitions, avoiding
    the O(n) indexing overhead of collections.deque when using random.sample.

    Attributes:
        capacity (int): Maximum number of transitions that can be stored.
        memory (list): Internal storage of transitions (pre-allocated with None).
        position (int): Index where the next transition will be written (circular).
        size (int): Current number of stored transitions (capped at capacity).
    """

    def __init__(self, capacity: int):
        """
        Initializes the replay memory with a given maximum capacity.

        Args:
            capacity: Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.memory = [None] * capacity   # Pre-allocate list for O(1) indexing
        self.position = 0                 # Next write index (circular)
        self.size = 0                     # Number of valid entries currently stored

    def push(self, *args) -> None:
        """
        Adds a new transition to the buffer.

        The arguments must match the fields of the Transition namedtuple:
        state, action, next_state, reward, next_action_mask.

        If the buffer is full, the oldest transition is overwritten.
        """
        # Create a Transition object and store it at the current position
        self.memory[self.position] = Transition(*args)

        # Advance the write pointer and wrap around if necessary
        self.position = (self.position + 1) % self.capacity

        # Keep track of the actual number of stored items (up to capacity)
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> list:
        """
        Returns a random batch of transitions of the requested size.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A list of Transition objects randomly selected from the buffer.
        """
        # Randomly choose indices from the valid range (0 .. size-1)
        indices = random.sample(range(self.size), batch_size)
        # List comprehension for O(1) random access into the list
        return [self.memory[i] for i in indices]

    def __len__(self) -> int:
        """
        Returns the current number of stored transitions.
        """
        return self.size