const { Feedforward, Population } = require('./index.js');

const and = [
    [[0, 0], [0]],
    [[1, 0], [0]],
    [[0, 1], [0]],
    [[1, 1], [1]],
];

const xor = [
    [[0, 0], [0]],
    [[1, 0], [1]],
    [[0, 1], [1]],
    [[1, 1], [0]],
];

function printOut(net, pattern) {
    pattern.forEach(p => {
        const res = net.map(p[0]);
        console.log(`  ${p[0]} -> ${p[1][0]} = ${res[0]} error: ${Math.abs(p[1][0] - res[0])}`);
    });
}

function trainNet(p) {
    console.log(`--- train - net [${p.name}] ---`);

    const net = new Feedforward({ model: p.model, flexibility: 1.0 });

    const measure = net.train({
        pattern: p.pattern,
        maxIterations: 120,
        minError: 0.1,
    });

    console.log(`iterations: ${measure.iterations} error: ${measure.error}`);

    printOut(net, p.pattern);
}

function trainPopulation(p) {
    console.log(`--- train - population [${p.name}] ---`);

    const pop = new Population({
        model: p.model,
        flexibility: 1.0,
    });

    const measure = pop.train({
        pattern: p.pattern,
        maxIterations: 120,
        minError: 0.1,
    });

    console.log(`iterations: ${measure.iterations} error: ${measure.error}`);

    printOut(measure.net, p.pattern);

    pop.nextGeneration();

    const measure2 = pop.train({
        pattern: p.pattern,
        maxIterations: 120,
        minError: 0.1,
    });

    console.log(`iterations: ${measure2.iterations} error: ${measure2.error}`);

    printOut(measure2.net, p.pattern);
}


[{
    name: 'and',
    pattern: and,
    model: [2, 1]
}, {
    name: 'xor',
    pattern: xor,
    model: [2, 2, 1]
}].forEach(p => {
    trainNet(p);
    trainPopulation(p);
})
