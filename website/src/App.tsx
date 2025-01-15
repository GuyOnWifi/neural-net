import React, { useEffect, useRef, useState } from 'react';
import network from "./assets/network.json";
import * as math from "mathjs";
import { Image, InterpolationAlgorithm } from "image-js";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

// @ts-ignore
import NeuralNet from "./neuralnet.js";

import './App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const options = {
  responsive: true,
  plugins: {
    legend: {
      position: 'top' as const,
    },
    title: {
      display: true,
      text: "Output probabilities",
    },
  },
};

const defaultData = {
  labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  datasets: [
    {
      label: 'Probabilities',
      data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      backgroundColor: 'rgba(53, 162, 235, 0.5)',
    },
  ],
};

function App() {
  const [data, setData] = useState(defaultData);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);
  const lastPosition = useRef({ x: 0, y: 0 });
  const neuralNet = useRef(null);
  
  const drawCharts = async(output : math.Matrix) => {
    
    let max = 0;
    let max_idx = 0;
    let values = [];

    for (let i = 0; i < 10; i++) {
      let val = output.get([i, 0]);
      values.push(val);
      if (val > max) {
        max = val;
        max_idx = i;
      }
    }
    console.log(max_idx);

    setData((d) => {
      const clone = structuredClone(d);
      clone.datasets[0].data = values;
      return clone;
    })

  }

  const evaluateDrawing = async () => {
    if (!canvasRef.current) return;
    let img = await Image.load(canvasRef.current.toDataURL());
    img = img.grey();
    img = img.resize({ width: 28, height: 28});
    
    let inp = math.matrix(Array.from(img.data));
    inp = math.dotDivide(inp, 255);
    inp = math.reshape(inp, [784, 1]);

    // @ts-ignore
    const output = neuralNet.current!.feedforward(inp);

    drawCharts(output);
  } 

  // Handle mouse down event to start drawing
  const startDrawing = (e : React.MouseEvent<HTMLCanvasElement>) => {
    isDrawing.current = true;
    const { offsetX, offsetY } = e.nativeEvent;
    lastPosition.current = { x: offsetX, y: offsetY };
  };

  // Handle mouse move event to draw lines
  const draw = (e : React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing.current) return;
    if (!canvasRef.current) return;

    const { offsetX, offsetY } = e.nativeEvent;
    const ctx = canvasRef.current.getContext('2d')!;
    const scale = document.getElementById("drawing-pad")?.clientWidth! / 28;


    ctx.beginPath();
    ctx.moveTo(lastPosition.current.x / scale, lastPosition.current.y / scale);
    ctx.lineTo(offsetX / scale, offsetY / scale);
    ctx.lineJoin = 'round';  // Smoother joints between lines
    ctx.lineCap = 'round';   // Smoother end points of lines
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    lastPosition.current = { x: offsetX, y: offsetY };
    evaluateDrawing();
  };

  // Handle mouse up event to stop drawing
  const stopDrawing = () => {
    isDrawing.current = false;
  };

  // Clear the canvas
  const clearCanvas = () => {
    if (!canvasRef.current) return;

    const ctx = canvasRef.current.getContext('2d')!;
    ctx.rect(0, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.fill();
  };

  useEffect(() => {
    clearCanvas();
    neuralNet.current = new NeuralNet(network);
    let a = math.random([784, 1]);
    // @ts-ignore
    let res = neuralNet.current.feedforward(a);
    console.log(res);
  }, [])

  useEffect(() => {
    console.log(data);
  }, [data])

  return (
    <div className="flex flex-row gap-4">
      <div className="flex flex-col gap-2 grow min-w-[500px] min-h-[500px]">
        <div id="drawing-pad" className="grow">
          <canvas
            ref={canvasRef}
            width={28}
            height={28}
            style={{ backgroundColor: 'black' }}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseOut={stopDrawing}
            className="w-[100%] border-2"
          />
        </div>
        <div onClick={clearCanvas} className="border-2 border-white cursor-pointer p-2 text-center">Clear</div>
      </div>
      <div className="border-white border-2 grow">
        <Bar options={options} data={data} />
      </div>
    </div>
  );
}

export default App;
