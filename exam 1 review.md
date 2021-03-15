### Characteristics of AI problems

- Knowledge often arrives recursively
- Problems exhibit recurring patterns
- Problems have multiple levels of granularity (粒度)
- Many problems are computationally intractable
- The world is dynamic, but knowledge of the world is static
- The world is open-ended, but knowledge is limited

### Characteristics of AI agents

- Agents have limited computing power. 

- Agents have limited sensors.
- Agents have limited attention.

- Computational logic is fundamentally deductive.

- AI agents’ knowledge is incomplete relative to the world.

### Difference between universal AI and specialized AI?

Universal AI is some goal we are working towards

Specialized AI is good at solving specific problems

## Semantic Networks

### Characteristics of good representations

- Make relationships explicit
- Expose natural constraints
- Bring objects and relations together
- Exclude extraneous details
- Transparent, concise, complete, fast, computable

## Generate and Test

### Dumb & Smart generators & testers

- dumb: generate anything that may happen with brute force, regardless of rules and productivity
- Meet the constraints
- be productive

FALSE: A smart tester does not generate duplicate states

## Means-Ends Analysis; Problem Reduction

### For each operator that can be applied

- Apply the operator to the current state
- calculate difference between new state and goal state
- Prefer state that minimizes distance between new state and goal state **(greedy)**

Literal distance (physical distance)

Figurative 比喻的 distance (distance between two states)

Means-Ends analysis can be applied to both distances

### How does problem reduction deal with the weakness of means-ends analysis?

Break the problem into two parts; sub-goals

## Production Systems

### Definition of cognitive agent

f: P* -> A (perception -> action)

### Three types of working memory

- procedural: How to do a certain thing? How to pour water into a glass? A list of instructions.

- semantic: What does the word mean? How does an airplane fly? Ideas and generalization; model to interpret certain things. 

- episodic: What is happening now and what should I do in such a condition? Events: last time I went to a good restaurant and I had a good time.

## Frames

### Properties of frames

- represent stereotypes

- Ate

   **subject** : Ashok

   **object** : a frog

   **location** : 

   **time** : 

   **utensils** : 

   **object-alive :** **false**

   **object-is** : **in-subject**

   **subject-mood** : **happy**

- provide default values

- Ate

   **subject** : Ashok

   **object** : a frog

   **location** : 

   **time** : 

   **utensils** : 

   **object-alive :** **false**

   **object-is** : **in-subject**

   **subject-mood** : **sad**

- exhibit inheritance

- Animal

   **#-of-legs :** 

   **#-of-arms :** 

- Ant (type of Animal)

   **#-of-legs :** **6**

   **#-of-arms :** **0**

**Method is not included in a frame**

## Recording cases

- Given problem A

- retrieve most similar prior problem B from memory
- apply B's solution to problem A

**Not good for a totally new problem.**

## Case-based reasoning

- **Retrieving** a case from memory similar to the current problem
- **Adapting** the solution to that case to fit the current problem
- **Evaluating** how well the adapted solution addresses the current problem
- **Storing** the new problem and solution as a case

### Assumptions

- Patterns exist in the world 

- Similar problems have similar solutions

## Incremental concept learning

- Is this an example of the concept?
  - Yes, but does not fit the current definition of the concept: **Generalize**
  - No, but it fit the current definition of the concept: **Specialize**

Over-generalization: Airplanes **are** birds because they have wings and can fly

Over-specialization: Bread+pork+Bread **is not** sandwich, because sandwiches must have lettuce, tomato and bacon

## Classification

Generalize percepts into concepts (equivalence classes) to avoid exponentially large space of percepts which make decisions hard

2^n percepts -> 2^m actions 

2^n percepts -> k concepts -> 2^m actions

## Tips

- strong words are probably wrong; pay more attention
- open-book! watch again the lecture video, refer to note, search online
  - not much time for searches
- 22 five-choice questions in 60 minutes. 1 to 4 correct options
- Honorlock: record the process

