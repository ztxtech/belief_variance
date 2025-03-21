
# Monte Carlo Experiment Visualization for Variance of Belief Information

This repository contains an interactive HTML visualization tool designed to simulate the distribution of belief masses and compute entropy/variance metrics based on the theoretical framework described in the paper ["Variance of Belief Information"](https://github.com/). The tool demonstrates how mass allocations affect entropy and variance dynamics in a two-element frame of discernment.

## Features
1. **Interactive Controls**: Adjust masses for red balls (m(R)), blue balls (m(B)), and indistinguishable balls (m({R,B}))
2. **Real-time Visualization**: Monte Carlo simulation of ball distributions with color-coded results
3. **Metric Calculation**: 
   - Deng Entropy (H) calculation using Eq.(7) from the paper
   - Belief Information Variance (V) calculation using Eq.(11)
4. **Markdown Documentation**: Built-in experiment description and parameter explanations

## Usage
1. **Adjust Parameters**:
   - **m(R)**: Slider controls the mass assigned to individual red balls
   - **m(B)**: Slider controls the mass assigned to individual blue balls
   - **Number of Experiments**: Adjust simulation iterations (1-10,000)
   
2. **Observe Results**:
   - Visual distribution of balls in the canvas
   - Real-time updates for:
     - Ball counts
     - m({R,B}) (computed as 1 - m(R) - m(B))
     - Deng Entropy (H)
     - Belief Information Variance (V)

## Theory Integration
- **Deng Entropy**: Measures uncertainty in mass distributions using Eq.(7)
- **Belief Variance**: Quantifies information content fluctuations using Eq.(11)
- **Mass Exchange**: Demonstrates variance behavior under different focal element allocations as discussed in Section 4.3

## Run the Experiment
1. Download the HTML file
2. Open directly in a web browser
3. Use sliders to modify parameters and observe results

## Dependencies
- Web browser with HTML5 Canvas support
- Marked.js for Markdown rendering (included via CDN)

## License
MIT License - see LICENSE file for details

This visualization tool supports the theoretical analysis presented in the paper by providing an interactive demonstration of key concepts. For detailed mathematical derivations, refer to the original publication.