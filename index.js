
class Feedforward {
    constructor(config) {
        this.config = Object.assign({
            model: [2, 1],
            flexibility: 0.5,
            activationThreshold: 0.85,
        }, config);

        this.sigs = [];
        this.errors = [];
        this.connections = [];
        this.momentums = [];

        this.config.model.forEach(m => {
            let arr = [];
            for (let i = 0; i < m; i++) {
                arr.push(0.0);
            }
            this.sigs.push(arr);
            this.errors.push(arr.slice());
        });

        for (let c = 0; c < this.config.model.length - 1; c++) {
            const s = this.config.model[c] + 1;
            const e = this.config.model[c + 1];

            let a = [];
            let b = [];
            for (let i = 0; i < (s * e); i++) {
                a.push(Math.random() * 2 - 1);
                b.push(0.0);
            }

            this.connections.push(a);
            this.momentums.push(b);
        }
    }

    static fromConnections(config, connections) {
        const net = new Feedforward(config);

        for (let c = 0; c < config.model.length - 1; c++) {
            const s = config.model[c] + 1;
            const e = config.model[c + 1];

            for (let i = 0; i < (s * e); i++) {
                net.connections[c][i] = connections[c][i];
            }
        }

        return net;
    }

    copy() {
        const copy = new Feedforward(this.config);

        for (let c = 0; c < this.config.model.length - 1; c++) {
            const s = this.config.model[c] + 1;
            const e = this.config.model[c + 1];

            for (let i = 0; i < (s * e); i++) {
                copy.connections[c][i] = this.connections[c][i];
            }
        }

        return copy;
    }

    map(input) {
        input.forEach((v, i) => {
            this.sigs[0][i] = v;
        });

        for (let layer = 1; layer < this.sigs.length; layer++) {
            const in_cells = this.sigs[layer - 1];
            const out_cells = this.sigs[layer];
            const conns = this.connections[layer - 1];

            for (let j = 0; j < out_cells.length; j++) {

                const offset = j * (in_cells.length + 1);
                let tmp = conns[offset + in_cells.length];

                for (let i = 0; i < in_cells.length; i++) {
                    tmp += in_cells[i] * conns[offset + i];
                }

                out_cells[j] = 1.0 / (1.0 + Math.exp(-2.0 * tmp));
            }
        }

        return this.sigs[this.sigs.length - 1].map(v => v > this.config.activationThreshold ? 1 : 0);
    }

    correct(output) {
        {
            const errors = this.errors[this.errors.length - 1];
            const sigs = this.sigs[this.sigs.length - 1];

            for (let i = 0; i < output.length; i++) {
                errors[i] = output[i] - (sigs[i] > this.config.activationThreshold ? 1.0 : 0.0);
            }
        }

        for (let j = this.errors.length - 1; j > 0; j--) {
            const in_err = this.errors[j];
            const in_cells = this.sigs[j];
            const out_err = this.errors[j - 1];
            const out_cells = this.sigs[j - 1];
            const out_count = out_err.length;
            const conns = this.connections[j - 1];
            const moms = this.momentums[j - 1];

            for (let o = 0; o < out_err.length; o++) {
                out_err[o] = 0.0;
            }

            for (let i = 0; i < in_err.length; i++) {
                const addeption = in_cells[i] * (1 - in_cells[i]) * in_err[i];
                const correction = addeption * this.config.flexibility;
                const offset = i * (out_count + 1);

                moms[offset + out_count] *= 0.5;
                conns[offset + out_count] += correction + moms[offset + out_count];
                moms[offset + out_count] += correction;
                for (let o = 0; o < out_count; o++) {
                    out_err[o] += addeption * conns[offset + o];

                    moms[offset + o] *= 0.5;
                    conns[offset + o] += out_cells[o] * correction + moms[offset + o];
                    moms[offset + o] += correction;
                }
            }
        }

    }

    train(config) {
        const outputCount = this.config.model[this.config.model.length - 1];
        let iterations = 0;
        let error = 0;

        for (let i = 0; i < config.maxIterations; i++) {
            iterations++;

            config.pattern.forEach(p => {
                this.map(p[0]);
                this.correct(p[1]);
            });

            error = config.pattern.reduce((e, p) => {
                return e + this.map(p[0]).reduce((x, v, i) => x + Math.abs(p[1][i] - v), 0);
            }, 0);

            error /= (config.pattern.length * outputCount);

            if (error <= config.minError) {
                break;
            }

        }

        return { iterations, error };
    }
}


class Population {
    constructor(config) {
        this.config = Object.assign({
            model: [2, 1],
            flexibility: 0.5,
            activationThreshold: 0.85,
            spread: 10
        }, config);
        this.nets = [];
        this.bestBuffer = null;

        for (let i = 0; i < this.config.spread; i++) {
            this.nets.push(new Feedforward(this.config));
        }
    }

    map(input) {
        return this.nets.map(net => net.map(input));
    }

    correct(output) {
        this.nets.forEach(net => net.correct(output));
    }

    train(options) {
        const outputCount = this.config.model[this.config.model.length - 1];
        let iterations = 0;
        let errors = [];

        for (let i = 0; i < options.maxIterations; i++) {
            iterations++;

            errors = this.nets.map(net => {
                options.pattern.forEach(p => {
                    net.map(p[0]);
                    net.correct(p[1]);
                });

                let error = options.pattern.reduce((e, p) => {
                    return e + net.map(p[0]).reduce((x, v, i) => x + Math.abs(p[1][i] - v), 0);
                }, 0);

                return { error: error / (options.pattern.length * outputCount), net };
            });

            if (errors.some(e => e.error <= options.minError)) {
                break;
            }
        }

        const best = errors.reduce((b, e) => b && b.error < e.error ? b : e, null);
        this.bestBuffer = best.net;
        return { iterations, error: best.error, net: best.net };
    }

    nextGeneration() {
        if (this.bestBuffer === null) {
            throw new Error('you have to train a population first!');
        }

        this.nets = [];

        for (let n = 0; n < this.config.spread - 1; n++) {
            const copy = new Feedforward(this.config);

            for (let c = 0; c < this.config.model.length - 1; c++) {
                const s = this.config.model[c] + 1;
                const e = this.config.model[c + 1];

                for (let i = 0; i < (s * e); i++) {
                    copy.connections[c][i] = this.bestBuffer.connections[c][i] + Math.random() - 0.5;
                }
            }

            this.nets.push(copy);
        }

        this.nets.push(this.bestBuffer);
        this.bestBuffer = null;
    }
}

exports.Feedforward = Feedforward;
exports.Population = Population;
