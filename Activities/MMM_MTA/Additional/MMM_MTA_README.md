# Marketing Mix Modeling (MMM) & Multi-Touch Attribution (MTA) Hands-On Lab

## Overview

This comprehensive Jupyter notebook provides a hands-on learning experience for understanding and implementing Marketing Mix Modeling (MMM) and Multi-Touch Attribution (MTA) - two essential techniques for measuring marketing effectiveness.

## What You'll Learn

### 1. Marketing Measurement Challenges
- Understanding multi-channel complexity
- Time lag effects (Adstock)
- Diminishing returns and saturation
- External factors affecting marketing performance

### 2. Marketing Mix Modeling (MMM)
- **What it is**: Top-down aggregate approach to measure media effectiveness
- **Core concepts**: Adstock transformation, saturation curves, base vs. incremental sales
- **Implementation**: Building MMM models with Ridge regression
- **Applications**: Budget allocation, channel ROI analysis, long-term planning
- **When to use**: Offline presence, strategic planning, privacy-first environments

### 3. Multi-Touch Attribution (MTA)
- **What it is**: Bottom-up customer journey analysis
- **Attribution models implemented**:
  - Last-Touch Attribution
  - First-Touch Attribution
  - Linear Attribution
  - Time-Decay Attribution
  - Position-Based (U-Shaped) Attribution
- **Applications**: Digital optimization, customer journey analysis, real-time decisions
- **When to use**: Digital-first businesses, short sales cycles, granular optimization

### 4. Comparing MMM vs MTA
- Side-by-side comparison of approaches
- When to use each method
- How to combine both for comprehensive insights
- Decision trees for selecting the right approach

### 5. Practical Considerations
- Common challenges and solutions
- Data quality requirements
- Model validation techniques
- Best practices for implementation
- Incrementality testing

### 6. Hands-On Exercises
- Budget optimization using MMM insights
- Comparing different attribution models
- Channel ROI calculation
- Real-world inspired datasets

## Dataset Information

The notebook uses **simulated real-world inspired datasets**:

### MMM Dataset
- **Type**: Retail/E-commerce marketing data
- **Time Period**: 2 years (104 weeks)
- **Channels**: TV, Digital, Radio, Print, Social Media
- **Metrics**: Weekly spend and sales
- **Realism**: Includes seasonality, holiday spikes, growth trends

### MTA Dataset
- **Type**: Customer journey clickstream data
- **Size**: 1,000 customer journeys
- **Touchpoints**: Social Media Ad, Search Ad, Display Ad, Email, Organic Search, Direct, Referral, Video Ad
- **Metrics**: Journey length, conversion value, touchpoint sequence
- **Realism**: Varying journey lengths (1-8 touchpoints), realistic first/last-touch distributions

### Why Simulated Data?

While we use simulated data in this notebook, it's designed to reflect real-world patterns:
- **Privacy**: No real customer data needed
- **Reproducibility**: Same results every time (random seed set)
- **Educational**: Clear cause-and-effect relationships for learning
- **Realistic**: Based on actual retail and e-commerce patterns

For production use, you would replace this with:
- Your actual marketing spend data (for MMM)
- Your clickstream/journey data (for MTA)

## Prerequisites

### Knowledge
- Basic understanding of Python programming
- Familiarity with pandas and numpy
- Basic statistics knowledge
- Understanding of marketing concepts

### Technical Requirements
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required packages (see requirements.txt):
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - scipy
  - statsmodels

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/tatwan/advanced_machine_learning.git
cd advanced_machine_learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Notebook

```bash
# Start Jupyter Notebook
jupyter notebook marketing_mix_modeling_and_attribution.ipynb

# Or use JupyterLab
jupyter lab marketing_mix_modeling_and_attribution.ipynb
```

### 3. Run the Cells

- Execute cells sequentially (Shift + Enter)
- Read the markdown explanations before each code section
- Experiment with parameters to see how results change

## Notebook Structure

The notebook is organized into 7 main sections:

1. **Introduction** (15 min)
   - Setup and library imports
   - Overview of marketing measurement challenges
   - Visual examples of multi-touch journeys and adstock effects

2. **MMM Deep Dive** (45 min)
   - Generate realistic marketing data
   - Implement adstock and saturation transformations
   - Build and train MMM model
   - Analyze channel contributions and ROI
   - Visualize model performance

3. **MTA Deep Dive** (45 min)
   - Generate customer journey data
   - Implement 5 attribution models
   - Compare attribution results
   - Visualize journey patterns and attribution differences

4. **MMM vs MTA Comparison** (20 min)
   - Side-by-side comparison matrix
   - Decision tree for selecting approach
   - Real-world use case examples

5. **Practical Considerations** (30 min)
   - Common challenges and solutions
   - Model diagnostics and validation
   - Statistical tests and assumptions
   - Best practices

6. **Hands-On Exercise** (30 min)
   - Budget optimization scenario
   - Channel reallocation based on ROI
   - Impact estimation
   - Visualization of recommendations

7. **Summary** (10 min)
   - Key takeaways
   - Implementation checklist
   - Common pitfalls to avoid
   - Resources for further learning

**Total Time**: 2.5 - 3 hours

## Key Features

### ✅ Comprehensive Coverage
- Both MMM and MTA covered in detail
- Theoretical foundations and practical implementation
- Real-world considerations and best practices

### ✅ Interactive Learning
- Fully executable code cells
- Visualizations throughout
- Hands-on exercises with solutions

### ✅ Production-Ready Code
- Well-structured classes for MMM and MTA
- Reusable functions and methods
- Comprehensive documentation and comments

### ✅ Realistic Examples
- Simulated data based on real-world patterns
- Industry-standard approaches
- Practical budget optimization scenarios

## Customization

### Using Your Own Data

#### For MMM:
Replace the data generation cell with:

```python
# Load your marketing data
mmm_data = pd.read_csv('your_marketing_data.csv')

# Required columns:
# - date (weekly/monthly timestamps)
# - channel_1_spend, channel_2_spend, ... (marketing spend by channel)
# - sales (or revenue, conversions, etc.)
```

#### For MTA:
Replace the journey generation cell with:

```python
# Load your customer journey data
journeys_df = pd.read_csv('your_journey_data.csv')

# Required columns:
# - customer_id
# - journey (list of touchpoints)
# - conversion_value
```

### Adjusting Parameters

Key parameters you can modify:

**MMM Model:**
- `adstock_decay`: Carryover effect (0-1). Higher = longer lasting impact
- `saturation_alpha`: Saturation scale parameter
- `saturation_gamma`: Saturation shape parameter. Lower = earlier saturation

**MTA Model:**
- `decay_rate`: How quickly credit decays for older touchpoints
- `first_last_weight`: Weight for first/last touch in position-based model
- Attribution window: Customize based on your sales cycle

## Validation and Testing

The notebook includes:
- Model diagnostics (residual analysis, Q-Q plots)
- Statistical tests (Shapiro-Wilk, Durbin-Watson)
- Train/test split validation
- Performance metrics (R², MAPE)

For production use, add:
- Incrementality tests (geo experiments, holdout tests)
- A/B tests for validation
- Regular model updates and monitoring

## Common Issues and Solutions

### Issue: Model overfits training data
**Solution**: Increase Ridge regularization parameter (alpha), use cross-validation, add more data

### Issue: Negative ROI for some channels
**Solution**: Check data quality, review transformations, consider interaction effects

### Issue: Residuals show patterns
**Solution**: Add more features (seasonality, events), try different transformations, check for missing variables

### Issue: Attribution models give very different results
**Solution**: This is normal - use multiple models, validate with incrementality tests, segment analysis

## Further Reading and Resources

### Books
- *Marketing Analytics* by Wayne Winston & Stephanie Goldsmith
- *Digital Marketing Analytics* by Chuck Hemann & Ken Burbary
- *Causal Inference: The Mixtape* by Scott Cunningham

### Papers
- "Challenges and Opportunities in Media Mix Modeling" (Google Research)
- "Multi-Touch Attribution in Online Advertising" (Various authors)

### Open Source Tools
- **Robyn** - Meta's open-source MMM framework
- **LightweightMMM** - Google's MMM implementation
- **PyMC-Marketing** - Bayesian MMM in Python

### Commercial Platforms
- Google Analytics 4 (GA4)
- Adobe Analytics
- Nielsen Marketing Cloud
- Neustar MarketShare

## Contributing

Found an issue or have a suggestion? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Contact the course maintainers

## License

This educational material is provided for learning purposes.

## Acknowledgments

- Based on industry best practices in marketing analytics
- Inspired by real-world implementations at leading companies
- Built with open-source tools and libraries

---

## Quick Start Summary

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Launch notebook**: `jupyter notebook marketing_mix_modeling_and_attribution.ipynb`
3. **Run all cells**: Kernel → Restart & Run All
4. **Duration**: 2.5-3 hours for full walkthrough
5. **Outcome**: Understanding of MMM and MTA with practical implementation skills

**Happy Learning! 🚀**
