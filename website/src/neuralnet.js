import * as math from "mathjs";


export default class NeuralNet {

    constructor(network) {
        this.sizes = network.sizes;
        
        this.weights = [];
        this.biases = [];
        
        for (let i = 0; i < network.weights.length; i++) {
            this.weights.push(math.matrix(network.weights[i]));
        }

        for (let i = 0; i < network.biases.length; i++) {
            this.biases.push(math.matrix(network.biases[i]));
        }

    }

    relu(v) {
        return math.map(v, (x) => math.max(x, 0));
    }
    
    softmax(v) {
        // for numerical stability
        const max = math.max(v);
        const exp = math.map(math.subtract(v, max), math.exp);
        return math.dotDivide(exp, math.sum(exp));
    }
    
    feedforward(v) {
        // all but last layer
        let i = 0;
        for (; i < this.sizes.length - 2; i++) {
            v = math.add(math.multiply(this.weights[i], v), this.biases[i]);
            v = this.relu(v);
        }
        // last softmax layer
        v = math.add(math.multiply(this.weights[i], v), this.biases[i]);
        v = this.softmax(v);

        return v;
    }
}


