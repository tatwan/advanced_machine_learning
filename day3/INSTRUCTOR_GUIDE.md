# Instructor Guide: Explainability and Regulation Lab

## Overview

This guide helps instructors effectively use the `explainability_regulation_lab.ipynb` in their teaching.

## Lab Objectives

Students will learn to:
1. Understand why model interpretability is essential in modern ML
2. Apply SHAP for global and local model explanations
3. Use LIME for quick local interpretability
4. Recognize regulatory requirements for AI systems
5. Detect and address bias in ML models
6. Generate compliant explanations for automated decisions

## Prerequisites

### Knowledge
- Basic machine learning concepts (classification, train/test split)
- Python programming (functions, data structures)
- Familiarity with pandas and scikit-learn
- Understanding of supervised learning

### Technical
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required packages (see requirements.txt)
- Internet connection (to download dataset)

## Time Estimates

| Activity | Duration |
|----------|----------|
| Self-paced reading + running cells | 60-90 min |
| With discussion and Q&A | 90-120 min |
| Including practice exercises | 2-3 hours |
| Full workshop format | 3-4 hours |

## Teaching Suggestions

### Part 1: Introduction (15-20 min)
**Cells 1-6**: Setup and Model Interpretability

**Discussion Points:**
- Why is explainability increasingly required by law?
- Real-world examples: loan denials, hiring, medical diagnosis
- The "accuracy vs. interpretability" tradeoff
- Recent AI regulation developments

**Activity:** Ask students to share examples where they'd want model explanations

### Part 2: Data and Model (15-20 min)
**Cells 7-9**: Dataset and Model Training

**Discussion Points:**
- Why the Adult Income dataset is relevant for this topic
- Sensitive attributes and their implications
- Model performance vs. fairness

**Activity:** Have students examine the dataset features and identify which might be "protected attributes"

### Part 3: SHAP (30-40 min)
**Cells 10-17**: SHAP Demonstrations

**Key Concepts to Emphasize:**
- SHAP is theoretically grounded (Shapley values)
- Difference between global and local explanations
- How to read SHAP visualizations (summary, force, dependence plots)
- SHAP values sum to model output

**Demonstration Tips:**
- Walk through one force plot in detail
- Show how SHAP values can be positive or negative
- Explain color coding in plots (red=high value, blue=low value)

**Common Student Questions:**
- "What's a SHAP value?" → Contribution of each feature to the prediction
- "Why use SHAP instead of feature importance?" → More accurate, handles interactions, local explanations
- "How long does it take?" → Fast for trees (TreeExplainer), slower for others

### Part 4: LIME (20-30 min)
**Cells 18-24**: LIME Demonstrations

**Key Concepts to Emphasize:**
- LIME creates a simple model to approximate complex model locally
- Model-agnostic (works with any model)
- Local fidelity vs. global accuracy
- May vary across runs (sampling-based)

**Comparison Activity:**
- Have students compare SHAP and LIME results for same prediction
- Discuss when results agree/disagree and why

### Part 5: Regulatory Considerations (30-40 min)
**Cells 25-31**: Compliance and Fairness

**Key Concepts to Emphasize:**
- Real regulations (GDPR, ECOA) require explainability
- Protected attributes and discrimination
- Statistical parity and other fairness metrics
- Adverse action notices are legally required

**Discussion Points:**
- Ethical considerations beyond legal requirements
- How to balance accuracy and fairness
- Human oversight in automated decisions

**Case Study Activity:**
- Walk through an adverse action notice
- Discuss how to communicate to non-technical stakeholders

### Part 6: Summary and Exercises (15-20 min)
**Cells 32-43**: Wrap-up

**Review:**
- When to use SHAP vs. LIME
- Key takeaways for regulatory compliance
- Best practices

## Common Issues and Solutions

### Issue: Dataset Won't Load
**Symptom:** Network error downloading from UCI
**Solution:** 
- Download manually from UCI repository
- Save as 'adult.data' locally
- Modify loading code to use local file

### Issue: SHAP/LIME Not Installed
**Symptom:** ImportError for shap or lime
**Solution:**
```bash
pip install shap lime
# Or for specific versions:
pip install shap==0.41.0 lime==0.2.0.1
```

### Issue: Visualizations Don't Display
**Symptom:** Plots not showing in notebook
**Solution:**
- Add `%matplotlib inline` at top of notebook
- Check if running in correct kernel
- Try restarting kernel

### Issue: Memory Error
**Symptom:** Kernel crashes or out of memory
**Solution:**
- Reduce sample size: `X_test_sample = X_test.sample(n=500)`
- Use fewer trees: `n_estimators=50`
- Clear unused variables
- Restart kernel between runs

### Issue: SHAP Takes Too Long
**Symptom:** SHAP computation hangs
**Solution:**
- Already using TreeExplainer (fast for Random Forest)
- Reduce sample size as above
- Use progress indicator to show it's working

## Assessment Ideas

### Formative Assessment (During Lab)
1. Ask students to interpret a SHAP force plot
2. Have them identify top 3 features from SHAP summary
3. Compare SHAP and LIME results for same prediction
4. Identify potential bias in fairness analysis

### Summative Assessment (After Lab)
1. Apply SHAP/LIME to a different dataset
2. Write adverse action notice for new scenario
3. Compare two models using explainability tools
4. Propose fairness improvements for biased model

### Discussion Questions
1. When would you choose SHAP over LIME, and vice versa?
2. How would you explain SHAP to a non-technical stakeholder?
3. What are the risks of relying solely on explainability tools?
4. How do regulations like GDPR affect ML system design?

## Extensions and Advanced Topics

### For Advanced Students
1. Implement SHAP interaction values
2. Compare multiple models (RF, XGBoost, Neural Network)
3. Explore different fairness metrics (equalized odds, etc.)
4. Create custom adverse action notice templates
5. Build a fairness-aware model

### Additional Datasets to Try
- Credit Approval dataset
- COMPAS recidivism dataset
- Medical diagnosis datasets (diabetes, heart disease)
- Custom domain-specific datasets

### Related Topics
- Fairlearn library for bias mitigation
- AI360 toolkit for comprehensive fairness
- Counterfactual explanations
- Model cards and documentation

## Key Takeaways for Students

✅ **Explainability is essential** - Not optional in many domains  
✅ **SHAP for thoroughness** - Use when you need rigorous explanations  
✅ **LIME for speed** - Use for quick local explanations  
✅ **Regulations are real** - GDPR, ECOA, and more require explanations  
✅ **Fairness matters** - Monitor and address bias proactively  
✅ **Communication is key** - Explain technical concepts to non-technical audiences  

## Resources for Instructors

### Papers
- "A Unified Approach to Interpreting Model Predictions" (SHAP paper)
- "Why Should I Trust You?" (LIME paper)
- "Fairness and Machine Learning" (Barocas et al.)

### Tools
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME GitHub](https://github.com/marcotcr/lime)
- [Fairlearn](https://fairlearn.org/)
- [AI Fairness 360](https://aif360.mybluemix.net/)

### Regulations
- [GDPR Article 22](https://gdpr-info.eu/art-22-gdpr/)
- [ECOA Overview](https://www.consumerfinance.gov/compliance/compliance-resources/lending-rules/equal-credit-opportunity-act/)
- [EU AI Act](https://artificialintelligenceact.eu/)

## Feedback and Improvements

We welcome feedback! Please note:
- What worked well
- What was confusing
- Time estimates vs. actual
- Additional topics to cover
- Technical issues encountered

## Version History

- v1.0 (Current): Initial release with Adult Income dataset
  - 43 cells (23 markdown, 20 code)
  - Complete SHAP and LIME coverage
  - Regulatory compliance section

---

**Questions?** Refer to `EXPLAINABILITY_LAB_README.md` for detailed usage instructions.
