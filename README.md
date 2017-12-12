# feedforward

**author:** Arne *(voidragon)* Simon [arne_simon@gmx.de]

A simple layered feedforward network with momentum based backward propergation.


**Please Donate**

+ **BTC:** 17voJDvueb7iZtcLRrLtq3dfQYBaSi2GsU
+ **ETC:** 0x7bC5Ff6Bc22B4C6Af135493E6a8a11A62D209ae5

## Example

### Using Feedforward

```js
const { Feedforward, Population } = require('feedforward');

const pattern = [
    [[0, 0], [0]],
    [[1, 0], [0]],
    [[0, 1], [0]],
    [[1, 1], [1]],
];

// --- train - feedforward ---

const net = new Feedforward({
    model: [2, 1],
    flexibility: 0.5
});

const measure = net.train({
    pattern,
    maxIterations: 70,
    minError: 0.1,
});

console.log(`iterations: ${measure.iterations} error: ${measure.error}`);

pattern.forEach(p => {
    const res = net.map(p[0]);
    console.log(`${p[1][0]} = ${res[0]} error: ${Math.abs(p[1][0] - res[0])}`);
});
```

### Using Population

```js
// --- train - population ---

const pop = new Population({
    model: [2, 1],
    spread: 10,
    flexibility: 0.5
});

const measure = pop.train({
    pattern,
    maxIterations: 70,
    minError: 0.1,
});

console.log(`iterations: ${measure.iterations} error: ${measure.error}`);

pattern.forEach(p => {
    const res = measure.net.map(p[0]);
    console.log(`${p[1][0]} = ${res[0]} error: ${Math.abs(p[1][0] - res[0])}`);
});
```

### Save and load a network

```js
// every thing that is needed is the config and the connections.
const data = JSON.stringify({
    config: net.config,
    connections: net.connections,
});

const loaded = JSON.parse(data);
const newNet = Feedforward.fromConnections(loaded.config, loaded.connections);
```

## API

### class Feedforward

A simple feedforward neuronal network.

#### fromConnections(config, connections)[static] -> net

Creates a new network for the config and initializes the connection values.

+ `config` - A feedforward config object.
+ `connections` - 2d array of connection values.
+ `net` - The newly created feedforward network.

#### new Feedforward(config)

+ `config`
    + `model` - Array of layer depths.
    + `flexibility` - How fast the net adepts.
    + `activationThreshold` - Idicates when an output cell identifies as on, aka 1.0 .

#### map(inputs) -> outputs

+ `inputs` - Array of input values, normalized between 1.0 and 0.0 .
+ `outputs` - An array of 0.0 or 1.0, indicating if the output neuron is off or on.

#### correct(outputs)

+ `outputs` - Array of expected output values, normalized between 1.0 and 0.0 .

#### train(options) -> measure

+ `options` - Training options.
    + `pattern` - The pattern to learn, aka approximate.
    + `maxInterations` - Maximum interations for the learning process.
    + `minError` - The minimum error at wich learning ends.
+ `measure` -
    + `error` - The avarage error for the pattern.
    + `iterations` - How many iterations where nessecary for learning the pattern.

### class Population

#### new Population(config)

+ `model` - Array of layer depths.
+ `spread` - How many networks will be generated.
+ `flexibility` - How fast the net adepts.
+ `activationThreshold` - Indicates when an output cell identifies as on, aka 1.0 .

#### map(inputs) -> outputs

+ `inputs` - Array of input values, normalized between 1.0 and 0.0 .

#### correct(outputs)

+ `outputs` - Array of expected output values, normalized between 1.0 and 0.0 .

#### train(options) -> measure

+ `options` - Training options.
    + `pattern` - The pattern to learn, aka approximate.
    + `maxInterations` - Maximum interations for the learning process.
    + `minError` - The minimum error at wich learning ends.
+ `measure` -
    + `error` - The avarage error for the pattern.
    + `iterations` - How many iterations where nessecary for learning the pattern.
    + `net` - The feedforward network with the lowest error.

#### nextGeneration()

Takes the best network from the last training and prodcues new population based on this network.
