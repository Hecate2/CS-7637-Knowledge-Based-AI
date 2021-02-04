### Consequence of moving A onto B:

- Allows the block below A to be moved
- prevents B to be moved
- the place of A has changed

### How to reach the goal?

- Blocks closer to the table must be placed with priority. How to compare the current state with the final goal? Use priority queues?
- To finally put A on B, we should first make A moveable, and clear everything on B (which also means making B moveable)